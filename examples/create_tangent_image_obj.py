import torch
import torch.nn.functional as F

import os

from tangent_images.util import *
from skimage import io

base_order = 1  # Determines the number of planes and their location on sphere
sample_order = 8  # Determines the sample resolution
scale_factor = 0.25  # How much to scale input image by

# Load and pre-process the image to donwsamples and convert to torch format
img = io.imread('earthmap4k.jpg')
img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
img = F.interpolate(
    img, scale_factor=scale_factor, mode='bilinear', align_corners=False)
img_shape = img.shape[-2:]

# ----------------------
# Create tangent images
# ----------------------

# Initialize layers (note: inputs need the batch dimension)
resample_to_uv_layer = ResampleToUVTexture(img_shape, base_order, sample_order)
resample_from_uv_layer = ResampleFromUVTexture(img_shape, base_order,
                                               sample_order)

# Put it on the GPU if possible
if torch.cuda.is_available():
    img = img.cuda()
    resample_to_uv_layer = resample_to_uv_layer.cuda()
    resample_from_uv_layer = resample_from_uv_layer.cuda()

# Resample the image to tangent planes (B x C x F_base x N x N)
patches = resample_to_uv_layer(img)

# Get rid of the batch dimension
patches.squeeze_(0)
N = patches.shape[-1]

# Write the patches to files
os.makedirs('patches', exist_ok=True)
for i in range(patches.shape[1]):
    patch = torch.flip(patches[:, i, ...], (1, ))
    io.imsave('patches/patch{:06d}.png'.format(i),
              patch.permute(1, 2, 0).byte().numpy())

# ----------------------
# Create mesh elements
# ----------------------

# Create base icosphere
icosphere = generate_icosphere(base_order)

# Number of samples in each dimension
num_samples = compute_num_samples(base_order, sample_order)
num_samples_with_pad = compute_num_samples(base_order, sample_order)

# Compute sampling resolutoin
sampling_resolution = get_sampling_resolution(base_order)

# Corners of tangent planes
corners = tangent_image_corners(icosphere, num_samples, num_samples,
                                sampling_resolution / num_samples,
                                sampling_resolution / num_samples, 'face')
corners = convert_spherical_to_3d(corners).squeeze()

# ------------------
# Write OBJ to file
# ------------------

# Write an OBJ file texturing the meshes in 3D
with open('textured_planes.obj', 'w') as f:
    f.write('mtllib textured_planes.mtl\n\n')

    for i in range(corners.view(-1, 3).shape[0]):
        f.write('v {} {} {}\n'.format(*(corners.view(-1, 3)[i, :] + 1)))

    f.write('\n')
    for i in range(corners.view(-1, 3).shape[0]):
        if i % 4 == 0:
            f.write('vt {} {}\n'.format(0.0, 0.0))
        elif i % 4 == 1:
            f.write('vt {} {}\n'.format(1.0, 0.0))
        elif i % 4 == 2:
            f.write('vt {} {}\n'.format(0.0, 1.0))
        elif i % 4 == 3:
            f.write('vt {} {}\n'.format(1.0, 1.0))

    for i in range(corners.shape[0]):
        f.write('\n')
        f.write('usemtl patches/patch{:06d}.png\n'.format(i))
        f.write('f {}/{} {}/{} {}/{}\n'.format(4 * i + 1, 4 * i + 1, 4 * i + 3,
                                               4 * i + 3, 4 * i + 4, 4 * i + 4))
        f.write('f {}/{} {}/{} {}/{}\n'.format(4 * i + 4, 4 * i + 4, 4 * i + 2,
                                               4 * i + 2, 4 * i + 1, 4 * i + 1))

with open('textured_planes.mtl', 'w') as f:
    for i in range(corners.shape[0]):
        f.write('newmtl patches/patch{:06d}.png\n'.format(i))
        f.write('map_Ka patches/patch{:06d}.png\n'.format(i))
        f.write('map_Kd patches/patch{:06d}.png\n\n'.format(i))