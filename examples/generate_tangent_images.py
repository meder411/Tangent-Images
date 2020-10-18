import os
import torch
import torch.nn.functional as F
from skimage import io
from spherical_distortion.util import load_torch_img, torch2numpy
from spherical_distortion.functional import create_tangent_images, tangent_images_to_equirectangular

base_order = 0  # Determines the number of planes and their location on sphere
sample_order = 9  # Determines the sample resolution
write_tan_imgs = True  # Whether to write all tan_imgs to images
write_back2img = True  # Whether to write the image resampled from patches
scale_factor = 1.0  # How much to scale input image by

# -------------------
# Load example image
# -------------------

# Load and pre-process the image to downsample and convert to torch format
os.makedirs('outputs', exist_ok=True)
img = load_torch_img('inputs/earthmap4k.jpg', True).float()
img = F.interpolate(img,
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=False,
                    recompute_scale_factor=True).squeeze(0)
img_shape = img.shape[-2:]

# Put it on the GPU if possible
if torch.cuda.is_available():
    img = img.cuda()

# -----------------------
# Create tangent images
# -----------------------

# Resample the image to tangent images (C x num_faces x N x N)
# Also create the valid-region masks. That is, the projection of the triangular face of the icosahedron onto the tangent image
tan_imgs, face_masks = create_tangent_images(img,
                                             base_order,
                                             sample_order,
                                             return_mask=True)

if write_tan_imgs:
    # Write the tangent images to files
    os.makedirs('outputs/tangent-images', exist_ok=True)
    for i in range(tan_imgs.shape[1]):
        patch = tan_imgs[:, i, ...]
        io.imsave('outputs/tangent-images/image{:06d}.png'.format(i),
                  torch2numpy(patch.byte()))
        mask = face_masks[i, ...]
        io.imsave('outputs/tangent-images/image{:06d}_mask.png'.format(i),
                  torch2numpy(255 * mask.byte()))

# -----------------------------------------
# Recreate original equirectangular image
# -----------------------------------------

if write_back2img:
    # Go from tangent images back to image
    back2img = tangent_images_to_equirectangular(tan_imgs, img_shape,
                                                 base_order, sample_order)

    # Write the image resampled back from tangent images (for sanity checking)
    io.imsave('outputs/back2img.png', torch2numpy(back2img.byte()))
