"""
EOT_Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torchvision.models as models

from tqdm import tqdm

from transforms.backpropable_transforms import sample_transformation

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class EOT_Net(nn.Module):
    def __init__(self, arch, use_eot, num_transforms, num_classes, max_offset_transforms=15):
        super(EOT_Net, self).__init__()
        
        assert arch in model_names, f"Unrecognized architecture: {arch}. I only know about {str(model_names)}"

        self.arch = arch
        self.use_eot = use_eot
        self.num_transforms = num_transforms
        self.num_classes = num_classes
        self.max_offset_transforms = max_offset_transforms

        print("=> Constructing EOT_Net with arch = '{}'".format(arch))
        model = models.__dict__[arch](pretrained=False)
        model.fc = torch.nn.Linear(512, num_classes)

        # DataParallel will divide and allocate batch_size to all available GPUs
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
        
        self.model = model
    
    def forward(self, batch, eot_override=None):
        if batch.is_cuda:
            assert False, "Batches should be on CPU before feeding into model"
        
        if eot_override == None:
            _use_eot = self.use_eot
        else:
            _use_eot = eot_override

        # Apply transformations
        if _use_eot:
            new_batch = torch.zeros(batch.shape[0] * self.num_transforms, batch.shape[1], batch.shape[2] + self.max_offset_transforms, batch.shape[3] + self.max_offset_transforms)
            index = 0
            for image in batch:
                for _ in range(self.num_transforms):
                    scale, T = sample_transformation(max_offset=self.max_offset_transforms)
                    new_batch[index] = T(image.clone())
                    index += 1
        else:
            new_batch = batch
        
        new_batch = new_batch.cuda()
        # print(new_batch.shape)

        if _use_eot:
            # Need to take the expectation over transforms here
            out = self.model(new_batch)
            out = torch.split(out, self.num_transforms, dim=0)

            avg_outs = []
            for elem in out:
                elem = F.log_softmax(elem, dim=1)
                avg_outs.append(torch.mean(elem, dim=0, keepdim=False))
            
            return torch.stack(avg_outs, dim=0)
        else:
            return self.model(new_batch)
