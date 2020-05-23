import torch
import torch.nn.functional as F
import os
from skimage import io

from spherical_distortion.functional import create_tangent_images, tangent_images_to_equirectangular
from spherical_distortion.util import *

# ----------------------
# Parameters
# ----------------------

base_order = 1  # Determines the number of planes and their location on sphere
sample_order = 8  # Determines the sample resolution
scale_factor = 0.25  # How much to scale input image by
skip = 1 # Images to skip in the OBJ (to only look at some tangent images)


# ----------------------
# Load equirectangular image
# ----------------------

# Load and pre-process the image to donwsamples and convert to torch format
img = load_torch_img('inputs/earthmap4k.jpg', True).float()
img = F.interpolate(
    img,
    scale_factor=scale_factor,
    mode='bilinear',
    align_corners=False,
    recompute_scale_factor=True).squeeze(0)
img_shape = img.shape[-2:]

# Put it on the GPU if possible
if torch.cuda.is_available():
    img = img.cuda()

# ----------------------
# Create tangent images
# ----------------------

# Resample the image to tangent images (B x C x num_faces x N x N)
tan_imgs = create_tangent_images(img, base_order, sample_order)

# Write the tangent images to files
os.makedirs('outputs/tangent-images', exist_ok=True)
for i in range(tan_imgs.shape[1]):
    patch = tan_imgs[:, i, ...]
    io.imsave('outputs/tangent-images/image{:06d}.png'.format(i),
              torch2numpy(patch.byte()))

# ----------------------
# Create mesh elements
# ----------------------

# Corners of tangent planes
corners = tangent_image_corners(base_order, sample_order)

# ------------------
# Write OBJ to file
# ------------------
# Write an OBJ file texturing the meshes in 3D
with open('outputs/textured_planes.obj', 'w') as f:
    f.write('mtllib textured_planes.mtl\n\n')

    for i in range(corners.view(-1, 3).shape[0]):
        f.write('v {} {} {}\n'.format(*(corners.view(-1, 3)[i, :] + 1)))

    f.write('\n')
    for i in range(corners.view(-1, 3).shape[0]):
        if i % 4 == 0:
            f.write('vt {} {}\n'.format(0.0, 1.0))
        elif i % 4 == 1:
            f.write('vt {} {}\n'.format(1.0, 1.0))
        elif i % 4 == 2:
            f.write('vt {} {}\n'.format(0.0, 0.0))
        elif i % 4 == 3:
            f.write('vt {} {}\n'.format(1.0, 0.0))

    for i in range(0, corners.shape[0], skip):
        f.write('\n')
        f.write('usemtl tangent-images/image{:06d}.png\n'.format(i))
        f.write('f {}/{} {}/{} {}/{}\n'.format(
            4 * i + 1, 4 * i + 1, 4 * i + 3, 4 * i + 3, 4 * i + 4, 4 * i + 4))
        f.write('f {}/{} {}/{} {}/{}\n'.format(
            4 * i + 4, 4 * i + 4, 4 * i + 2, 4 * i + 2, 4 * i + 1, 4 * i + 1))

with open('outputs/textured_planes.mtl', 'w') as f:
    for i in range(0, corners.shape[0], skip):
        f.write('newmtl tangent-images/image{:06d}.png\n'.format(i))
        f.write('map_Ka tangent-images/image{:06d}.png\n'.format(i))
        f.write('map_Kd tangent-images/image{:06d}.png\n\n'.format(i))