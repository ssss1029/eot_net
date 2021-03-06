import random
import torch 

import logging
import functools
import inspect

import numpy as np

import torchvision

# Useful for undoing thetorchvision.transforms.Normalize() 
# From https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # The normalize code -> t.sub_(m).div_(s)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

"""
All transforms here are backprop-able. They expect PyTorch tensors and return PyTorch tensors
    Input:  C x H x W
    Output: C X H X W
"""

def Grayscale(tensor):
    num_channels = tensor.shape[0]
    mean = torch.mean(tensor, dim=0, keepdim=True)
    tensor = torch.cat([mean for _ in range(num_channels)], dim=0)
    return tensor

def HorizontalFlip(tensor):
    tensor = torch.flip(tensor, dims=[2])
    return tensor

def VerticalFlip(tensor):
    tensor = torch.flip(tensor, dims=[1])
    return tensor

def TranslateX(tensor, shift_x):
        IMG_SIZE = tensor.shape[1]

        out = torch.ones_like(tensor)

        # Handle x-shift
        if shift_x >= 0:
            out[:,:, shift_x:] = tensor[:,:, :IMG_SIZE - shift_x]
            out[:,:, :shift_x] = 0
        else:
            out[:,:, :shift_x] = tensor[:, :, -shift_x:]
            out[:,:, shift_x:] = 0
        
        return out

def TranslateY(tensor, shift_y):
        IMG_SIZE = tensor.shape[1]
        
        out = torch.ones_like(tensor)

        # Handle y-shift
        if shift_y >= 0:
            out[:, shift_y:] = tensor[:, :IMG_SIZE - shift_y]
            out[:, :shift_y] = 0
        else:
            out[:, :shift_y] = tensor[:, -shift_y:]
            out[:, shift_y:] = 0
        return out

def ColorShift(tensor, shift_c1, shift_c2, shift_c3):
    tensor[0, :, :] *= shift_c1
    tensor[1, :, :] *= shift_c2
    tensor[2, :, :] *= shift_c3
    return tensor

def Rotate(tensor, k):
    return torch.rot90(tensor, k=k, dims=[1, 2])

def PermuteColorChannels(tensor, color_channels):
    return tensor[color_channels, :, :]

def MirrorPadOffset(tensor, dx, dy, max_offset):
    """
    From E-LPIPS
    """
    orig_size = tensor.shape[2]
    pad_bottom = max_offset - dy
    pad_left   = max_offset - dx
    pad_top    = dy
    pad_right  = dx

    rpad = torch.nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))

    tensor = tensor.unsqueeze(0)
    padded = rpad(tensor).squeeze(0)  # (C, H + max_offset, W + max_offset)
    return padded

def DownscaleBox(tensor, scale):
    if scale == 1:
        return tensor
    
    C, H, W = tensor.shape[0], tensor.shape[1], tensor.shape[2]

    res = torch.zeros((C, H, W))

    assert H % scale == 0
    assert W % scale == 0
    assert H == W

    new_height = H // scale
    new_width = W // scale

    tensor = torch.reshape(tensor, (C, new_height, scale, new_width, scale))
    tensor = torch.mean(tensor, dim=[2, 4])

    top_index = random.randint(0, H - new_height - 1)
    left_index = random.randint(0, W - new_width - 1)
    res[:, top_index:(top_index + new_height), left_index:(left_index + new_width)] = tensor
    return res

def SwapXY(tensor):
    return tensor.permute(0, 2, 1)

def sample_transformation(max_offset=15):
    dx, dy = random.randint(0, max_offset), random.randint(0, max_offset)
    
    flip_x, flip_y, swap_xy = random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)
    
    shift_r, shift_g, shift_b = random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), random.uniform(0.2, 1.0)
    
    rotate_k = random.randint(0, 3)

    permute_color_channels_order = [0, 1, 2]
    random.shuffle(permute_color_channels_order)

    choices = np.array([1, 2])
    probs = (1 / (choices ** 2)) / (np.sum(1 / (choices ** 2)))
    scale = np.random.choice(
        choices,
        p=probs
    )
    # scale_dx = random.randint(0, scale - 1)
    # scale_dy = random.randint(0, scale - 1)

    def transform(tensor):
        # tensor = TranslateX(tensor, shift_x=dx)
        # tensor = TranslateY(tensor, shift_y=dy)
        tensor = MirrorPadOffset(tensor, dx, dy, max_offset)
        tensor = DownscaleBox(tensor, scale)
        if flip_x == 1:
            tensor = HorizontalFlip(tensor)
        if flip_y == 1:
            tensor = VerticalFlip(tensor)
        if swap_xy == 1:
            tensor = SwapXY(tensor)
        tensor = ColorShift(tensor, shift_c1=shift_r, shift_c2=shift_g, shift_c3=shift_b)
        tensor = Rotate(tensor, k=rotate_k)
        # tensor = PermuteColorChannels(tensor, color_channels=permute_color_channels_order)
        return tensor

    return scale, transform

class BPTransform(object):

    def __init__(self, max_offset=None):
        self.max_offset = max_offset
        pass

    def __call__(self, sample):
        if self.max_offset:
            scale, transform = sample_transformation(max_offset = self.max_offset)
        else:
            scale, transform = sample_transformation()
        return transform(sample)

if __name__ == "__main__":
    from PIL import Image
    import torchvision
    import torchvision.transforms as transforms


    preprocess_transform = transforms.Compose([
        transforms.Scale(185),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    import backpropable_transforms as bptransforms
    unnormalize = bptransforms.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    img_x = preprocess_transform(Image.open("/accounts/projects/jsteinhardt/sauravkadavath/eot_net/deepika.png").convert('RGB'))

    transform = sample_transformation()

    img_x = transform(img_x)

    torchvision.utils.save_image(img_x, "/accounts/projects/jsteinhardt/sauravkadavath/eot_net/sample_transformation.png")
