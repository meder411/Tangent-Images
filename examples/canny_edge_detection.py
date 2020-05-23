import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage import io, feature, color
from spherical_distortion.util import load_numpy_img, torch2numpy, numpy2torch
from spherical_distortion.functional import create_tangent_images, tangent_images_to_equirectangular

base_order = 0  # Determines the number of planes and their location on sphere
sample_order = 10  # Determines the input spherical resolution
sigma = 1.0  # Canny edge detection sigma value
scale_factor = 1.0  # How much to scale input image by

# -------------------
# Load example image
# -------------------

# Load and pre-process the image to donwsamples and convert to torch format
os.makedirs('outputs', exist_ok=True)
img = load_numpy_img('inputs/stanford-example.png')[..., :3]
img = color.rgb2gray(img)[..., None]  # Convert to grayscale
img = numpy2torch(img).float()  # Convert to torch
img = F.interpolate(
    img.unsqueeze(0),
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
tan_imgs = create_tangent_images(img, base_order, sample_order)

# Compute the Canny detector on each tangent image
tan_edges = torch.zeros_like(tan_imgs)
for i in range(tan_imgs.shape[1]):
    tan_edges[0, i, ...] = numpy2torch(
        feature.canny(torch2numpy(tan_imgs[0, i, ...]), sigma=sigma))

# -----------------------------------------
# Recreate original equirectangular image
# -----------------------------------------
# Resample tangent images back to equirectangular image
edges_from_tan_imgs = tangent_images_to_equirectangular(
    tan_edges, img_shape, base_order, sample_order)
io.imsave('outputs/canny_edges_b{}.png'.format(base_order),
          (255 * torch2numpy(edges_from_tan_imgs)).astype(np.uint8))

# Also run the detector on the equirectangular image and save the output
edges_from_erp = feature.canny(torch2numpy(img)[..., 0], sigma=sigma)
io.imsave('outputs/canny_edges_erp.png', (255 * edges_from_erp).astype(
    np.uint8))