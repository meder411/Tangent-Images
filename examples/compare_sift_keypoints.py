import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images, unresample
from spherical_distortion.util import *
import matplotlib.pyplot as plt
from skimage import io
import os

# ----------------------------------------------
# Parameters
# ----------------------------------------------
base_order = 1  # Base sphere resolution
sample_order = 10  # Determines sample resolution (10 = 2048 x 4096)
scale_factor = 1.0  # How much to scale input equirectangular image by
save_ply = False  # Whether to save the PLY visualizations too

# ----------------------------------------------
# Compute necessary data
# ----------------------------------------------
# Corners of tangent planes in spherical coordinates (N x 4 x 3)
corners = tangent_image_corners(base_order, sample_order)

# ----------------------------------------------
# Load and preprocess the image
# ----------------------------------------------
os.makedirs('outputs', exist_ok=True)
img = load_torch_img('inputs/stanford-example2.png')[:3, ...].float()
img = F.interpolate(img.unsqueeze(0),
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=False,
                    recompute_scale_factor=True).squeeze(0)

# Resample the image to N tangent images (out: 3 x N x H x W)
tex_image = create_tangent_images(img, base_order, sample_order).byte()

if save_ply:
    # Create and save textured sphere at sample order
    viz_icosphere = generate_icosphere(sample_order)
    sample_map = equirectangular_to_sphere_resample_map(
        img.shape[-2:], viz_icosphere)
    rgb_vertices = unresample(img, sample_map,
                              InterpolationType.BISPHERICAL).squeeze()
    viz_icosphere.normalize_points()
    write_ply('outputs/textured_sphere.ply',
              viz_icosphere.get_vertices().transpose(0, 1).numpy(),
              rgb=rgb_vertices.cpu().numpy(),
              faces=viz_icosphere.get_all_face_vertex_indices().T.numpy(),
              text=False)

# -------------------------------------------------
# Compute SIFT descriptors for each tangent image
# -------------------------------------------------
tangent_image_kp, tangent_image_desc = sift_tangent_images(tex_image,
                                                           base_order,
                                                           sample_order,
                                                           img.shape[-2:],
                                                           crop_degree=25)
print('Num Tangent Image Keypoints:', tangent_image_kp.shape[0])

if save_ply:
    # Write tangent image keypoints to their 3D position on the sphere
    kp_3d = convert_spherical_to_3d(
        convert_image_to_spherical(tangent_image_kp[:, :2], img.shape[-2:]))

    # Write 3D keypoints
    write_ply('outputs/kp_tangent_images.ply', kp_3d.view(-1, 3).numpy().T)

# ---------------------------------------------------
# Compute SIFT descriptors on equirectangular image
# ---------------------------------------------------
erp_kp_details = sift_equirectangular(img, crop_degree=25)
erp_kp = erp_kp_details[0]
erp_desc = erp_kp_details[1]
print('Num Equirect. Keypoints:', erp_kp.shape[0])
if save_ply:
    kp_3d = convert_spherical_to_3d(
        convert_image_to_spherical(erp_kp[:, :2], img.shape[-2:]))
    write_ply('outputs/kp_equirectangular.ply', kp_3d.view(-1, 3).numpy().T)

# ----------------------------------------------
# Plot descriptors
# ----------------------------------------------

# Convert image back to numpy format
img = torch2numpy(img.byte())

# Create a figure and axis handle
fig, ax = plt.subplots(1, 1)

# Set up the plot
ax.set_aspect(1, adjustable='box')
ax.imshow(img)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.plot(erp_kp[:, 0], erp_kp[:, 1], 'r.', markersize=0.7)
tangent_image_kp[(tangent_image_kp[:, [0]] >= img.shape[-2] -
                  1).expand_as(tangent_image_kp)] = float('nan')
ax.plot(tangent_image_kp[:, 0], tangent_image_kp[:, 1], 'b.', markersize=0.7)
plt.axis('off')

fig.savefig('outputs/sift-keypoints.png',
            bbox_inches='tight',
            pad_inches=0,
            dpi=600)
plt.show()