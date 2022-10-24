"""
This file introduces how to generate high resolution AG-MNIST samples
"""
import torch
import torchvision
import os
from PIL import Image
import random
from torchvision import datasets, transforms


def ag_corrupt_high_resolution(imgs, threshold=0.5, interval=8, direction=(1,0)):
    '''
    Apply AG corruption on MNIST samples (1x28x28) to generate high resolution AG-MNIST samples(3x224x224)
    params:
        imgs: High resolution MNIST samples to corrupt, must have the shape B,C,W,H.
        threshold: Threshold to separate the figure and the background.
        interval: The interval between abutting gratings.
        direction: Direction of abutting gratings. 
                   (1,0) : horizontal
                   (0,1) : vertical
                   (1,1) : upper right to lower left
                   (-1,1) : upper left to lower right
    '''
    assert len(imgs.shape) == 4, "The images must have four dimensions of B,C,W,H."   
    imgs = torch.nn.functional.interpolate(imgs, scale_factor = 8, mode = 'bilinear', align_corners = False)
    imgs = torch.cat([imgs, imgs, imgs], dim=1)
    #return imgs
    B,C,W,H = imgs.shape
    mask_fg = (imgs>threshold).float()  
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
  
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==interval//2:
                gratings_bg[:,:,w,h] = 1
    masked_gratings_fg = mask_fg*gratings_fg
    masked_gratings_bg = mask_bg*gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg
    return ag_image


transform = transforms.Compose([transforms.ToTensor()])
test_set = datasets.MNIST('./raw_datasets/', train=False, transform = transform, download=False)
images, labels = test_set[0]
images = images.unsqueeze(0)

ag_images = ag_corrupt_high_resolution(images, threshold=0.5, interval=8, direction=(1,0))  # alter the interval and direction
torchvision.utils.save_image(ag_images[0], 'high_resolution_ag_mnist_sample.png')
