import os
import random
import shutil
import time
import warnings
import math
import numpy as np
from PIL import ImageOps, Image
import tempfile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm

import models.eot_net as eot_net
from transforms.backpropable_transforms import BPTransform, sample_transformation, UnNormalize


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default="saved_images/attack_eot_1/")
parser.add_argument("--net-model-path", type=str, default=None)
parser.add_argument('--num-transforms', type=int, default=10)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

# 200 classes used in ImageNet-R
imagenet_r_wnids = ['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']
imagenet_r_wnids.sort()
classes_chosen = imagenet_r_wnids[::2] # Choose 40 classes for our dataset
assert len(classes_chosen) == 100

class ImageNetSubsetDataset(datasets.ImageFolder):
    """
    Dataset class to take a specified subset of some larger dataset
    """
    def __init__(self, root, *args, **kwargs):
        
        print("Using {0} classes {1}".format(len(classes_chosen), classes_chosen))

        self.new_root = tempfile.mkdtemp()
        for _class in classes_chosen:
            orig_dir = os.path.join(root, _class)
            assert os.path.isdir(orig_dir)

            os.symlink(orig_dir, os.path.join(self.new_root, _class))
        
        super().__init__(self.new_root, *args, **kwargs)
    
    def __del__(self):
        # Clean up
        shutil.rmtree(self.new_root)

if args.num_transforms == 0:
    model = eot_net.EOT_Net(
        arch='resnet18', 
        use_eot=False, 
        num_transforms=None, 
        num_classes=100, 
        max_offset_transforms=None
    )
else:
    model = eot_net.EOT_Net(
        arch='resnet18', 
        use_eot=True, 
        num_transforms=args.num_transforms, 
        num_classes=100, 
        max_offset_transforms=15
    )

model.load_state_dict(torch.load(args.net_model_path)['state_dict'])
model.eval()
model.cuda()

unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    ImageNetSubsetDataset(
        '/var/tmp/namespace/hendrycks/imagenet/val', 
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224 - 15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    ),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True
)

def generate_adv(model, image, target):
    print("Real target = ", target)
    
    with torch.no_grad():
        save_image(unnormalize(image.clone().unsqueeze(0)), os.path.join(args.save_dir, "original_image.png"))

    image_original = image.clone().detach()
    
    image.requires_grad = True
    optimizer = torch.optim.Adam([image], lr=args.lr)

    for i in range(args.iterations):
        logits = model(image.clone().unsqueeze(0))
        print(torch.nn.functional.softmax(logits))

        # Non-targeted adversarial attack loss
        # loss = -1.0 *  F.nll_loss(logits, target.unsqueeze(0))

        # Targeted loss to burrito
        loss = F.nll_loss(logits, torch.tensor([97]).cuda())

        loss.backward()

        optimizer.step()
        image.data = tensor_clamp_l2(image.data, image_original.data, args.epsilon)
        
        print(">>>>>> Iteration", i)
        print("loss = ", loss.item())
        print("|| x - x_original || = ", torch.sqrt(torch.sum((image - image_original) ** 2)).item())
        print("------------------------------------")

        with torch.no_grad():
            save_image(unnormalize(image.clone()), os.path.join(args.save_dir, f"perturbed_images/image_{i}.png"))
            save_image(image - image_original + 0.5, os.path.join(args.save_dir, f"differences/image_{i}.png"))

def main():
    # switch to evaluate mode
    model.eval()

    end = time.time()
    C = 0
    for i, (images, targets) in enumerate(val_loader):
        targets = targets.cuda()

        for k in range(images.shape[0]):
            C += 1

            if C == 2:
                generate_adv(model, images[k], targets[k])
                
                print("Done with one adv example")
                exit()

def tensor_clamp_l2(x, center, radius):
    diff = x - center
    diff_norm = torch.sqrt(torch.sum(diff ** 2)).item()
    if diff_norm > radius:
        ret = (center + (diff / diff_norm) * radius)
        return ret
    else:
        return x

main()