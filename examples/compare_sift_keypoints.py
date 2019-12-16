import torch
import torch.nn.functional as F
from tangent_images.util import *
from tangent_images.nn import *
import matplotlib.pyplot as plt
from skimage import io, transform

# ----------------------------------------------
# Parameters
# ----------------------------------------------
base_order = 1  # Base sphere resolution
sample_order = 10  # Determines sample resolution (10 = 2048 x 4096)
scale_factor = 1  # How much to scale input equirectangular image by
save_ply = True  # Whether to save the PLY visualizations too

# ----------------------------------------------
# Compute necessary data
# ----------------------------------------------
# Generate the base icosphere
icosphere = generate_icosphere(base_order)

# Number of samples per tangent image dimension
num_samples = compute_num_samples(base_order, sample_order)

# Sampling resolutions
sampling_resolution = get_sampling_resolution(base_order)

# Corners of tangent planes in spherical coordinates (N x 4 x 3)
corners = tangent_image_corners(
    icosphere, num_samples, num_samples, sampling_resolution / num_samples,
    sampling_resolution / num_samples, 'face').squeeze(0)

# Convert the spherical corners to corners in 3D space
corners = convert_spherical_to_3d(corners)

# ----------------------------------------------
# Load and preprocess the image
# ----------------------------------------------
img = io.imread('earthmap4k.jpg').astype(np.float32)
img = (255 * transform.rescale(
    img / 255,
    scale=scale_factor,
    order=1,
    anti_aliasing=True,
    mode='constant',
    multichannel=True)).astype(np.uint8)

# Convert image to torch format
img = torch.from_numpy(img).float().permute(2, 0, 1).contiguous()

# Module to resample the image to UV textures
resample_to_uv_layer = ResampleToUVTexture(img.shape[-2:], base_order,
                                           sample_order, 1)

# Resample the image to N tangent images (out: 3 x N x H x W)
tex_image = resample_to_uv_layer(img.unsqueeze(0)).squeeze(0).byte()

if save_ply:
    # Create and save textured sphere at sample order
    viz_icosphere = generate_icosphere(sample_order)
    unresampler = Unresample('bispherical')
    sample_map = equirectangular_to_sphere_resample_map(img.shape[-2:],
                                                        viz_icosphere)
    rgb_vertices = unresampler(img.unsqueeze(0), sample_map).squeeze()
    viz_icosphere.normalize_points()
    write_ply(
        'textured_sphere.ply',
        viz_icosphere.get_vertices().transpose(0, 1).numpy(),
        rgb=rgb_vertices.cpu().numpy(),
        faces=viz_icosphere.get_all_face_vertex_indices().numpy(),
        text=False)

# ----------------------------------------------
# Compute SIFT descriptors for each patch
# ----------------------------------------------
tangent_image_kp, tangent_image_desc = sift_tangent_images(
    tex_image, corners, img.shape[-2:], crop_degree=0)
print('Num Tangent Image Keypoints:', tangent_image_kp.shape[0])

if save_ply:
    # Write tangent image keypoints to their 3D position on the sphere
    kp_3d = convert_spherical_to_3d(
        convert_image_to_spherical(tangent_image_kp[:, :2], img.shape[-2:]))

    # Write 3D keypoints
    write_ply('kp_tangent_images.ply', kp_3d.view(-1, 3).numpy().T)

# ---------------------------------------------------
# Compute SIFT descriptors on equirectangular image
# ---------------------------------------------------
erp_kp_details = sift_equirectangular(img, crop_degree=0)
erp_kp = erp_kp_details[0]
erp_desc = erp_kp_details[1]
print('Num Equirect. Keypoints:', erp_kp.shape[0])
if save_ply:
    kp_3d = convert_spherical_to_3d(
        convert_image_to_spherical(erp_kp[:, :2], img.shape[-2:]))
    write_ply('kp_equirectangular.ply', kp_3d.view(-1, 3).numpy().T)

# ----------------------------------------------
# Plot descriptors
# ----------------------------------------------

# Convert image back to numpy format
img = img.permute(1, 2, 0).byte().numpy()

# Create a figure and axis handle
fig, ax = plt.subplots(1, 1)

# Set up the plot
ax.set_aspect(1, adjustable='box')
ax.imshow(img)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.plot(erp_kp[:, 0], erp_kp[:, 1], 'r.', markersize=3.0)
ax.plot(tangent_image_kp[:, 0], tangent_image_kp[:, 1], 'b.', markersize=3.0)
plt.show()