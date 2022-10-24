"""
This file introduces how to generate AG-silhouette samples
"""
import torch
import torchvision
import os
from PIL import Image
import random
from torchvision import datasets, transforms


def ag_corrupt_silhouettes(imgs, threshold=0.5, interval=2, direction=(1,0)):
    '''
    Apply AG corruption on silhouette samples (224x224)
    params:
        imgs: silhouette samples to corrupt, must have the shape B,C,W,H.
        threshold: Threshold to separate the figure and the background.
        interval: The interval between abutting gratings.
        direction: Direction of abutting gratings. 
                   (1,0) : horizontal
                   (0,1) : vertical
                   (1,1) : upper right to lower left
                   (-1,1) : upper left to lower right
    '''
    assert len(imgs.shape) == 4, "The images must have four dimensions of B,C,W,H."
    #imgs = torch.nn.functional.interpolate(imgs, scale_factor = 2, mode = 'bilinear', align_corners = False)
    B,C,W,H = imgs.shape
    mask_fg = (imgs<threshold).float()
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==interval//2:
                gratings_bg[:,:,w,h] = 1
    ag_images = mask_fg*gratings_fg + mask_bg*gratings_bg
    return ag_images



transform = transforms.Compose([transforms.ToTensor()])
img = Image.open("./raw_datasets/silhouettes/airplane/airplane1.png")
img = transform(img).unsqueeze(0) # expand the dimension of the image to 
ag_images = ag_corrupt_silhouettes(img, threshold=0.5, interval=8, direction=(1,0))    
torchvision.utils.save_image(ag_images, 'ag_silhouette_sample.png')