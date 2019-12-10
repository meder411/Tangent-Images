import torch
import torch.nn.functional as F

import os

from tangent_images.util import *
from skimage import io

base_order = 1  # Determines the number of planes and their location on sphere
sample_order = 10  # Determines the sample resolution
kernel_size = 1  # Determines the padding to sample
write_patches = False  # Whether to write all patches to images
write_back2img = True  # Whether to write the image resmapled from patchs
scale_factor = 1.0  # How much to scale input image by

# Load and pre-process the image to donwsamples and convert to torch format
img = io.imread('earthmap4k.jpg')
img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
img = F.interpolate(img,
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=False)
img_shape = img.shape[-2:]
channels = img.shape[1]

resample_to_uv_layer = ResampleToUVTexture(img_shape, base_order, sample_order,
                                           kernel_size)
resample_from_uv_layer = ResampleFromUVTexture(img_shape, base_order,
                                               sample_order)

# Put it on the GPU if possible
if torch.cuda.is_available():
    img = img.cuda()
    resample_to_uv_layer = resample_to_uv_layer.cuda()
    resample_from_uv_layer = resample_from_uv_layer.cuda()

# Resample the image to tangent planes (B x C x F_base x N x N)
patches = resample_to_uv_layer(img)

# Go from patches back to image
back2img = resample_from_uv_layer(patches)

# Get rid of the batch dimension
patches.squeeze_(0)

# Write the patches to files
if write_patches:
    N = patches.shape[-1]
    os.makedirs('patches', exist_ok=True)
    for i in range(patches.shape[1]):
        patch = torch.flip(patches[:, i, ...], (1, ))
        io.imsave('patches/patch{:06d}.png'.format(i),
                  patch.permute(1, 2, 0).byte().numpy())

# Write the image resmapled back from patches
if write_back2img:
    io.imsave('back2img.png',
              back2img.squeeze(0).permute(1, 2, 0).byte().numpy())
