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
sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
# Number of samples in each dimension
# 2^(difference between sample order and base sphere order) + padding
num_samples = 2**(sample_order - base_order)
scale_factor = 0.25
save_ply = True
write_openmvg_files = True

# ----------------------------------------------
# Compute necessary data
# ----------------------------------------------
# Generate the base icosphere
icosphere = generate_icosphere(base_order)

# This might come in handy later
num_faces = compute_num_faces(base_order)

# After level 4, the vertex resolution comes pretty close to exactly halving at each subsequent order. This means we don't need to generate the sphere to compute the resolution. However, at lower levels of subdivision, we ought to compute the vertex resolution as it's not fixed
if base_order < 5:
    sampling_resolution = generate_icosphere(max(
        0, base_order - 1)).get_vertex_resolution()
    if base_order == 0:
        sampling_resolution *= 2
else:
    sampling_resolution = generate_icosphere(5 - 1).get_vertex_resolution()
    sampling_resolution /= (2**(base_order - 5))

# Corners of tangent planes
corners = tangent_plane_corners(icosphere, num_samples, num_samples,
                                sampling_resolution / num_samples,
                                sampling_resolution / num_samples, 'face')
corners = convert_spherical_to_3d(corners).squeeze()

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

# Module to resample the image to UV textures
resample_to_uv_layer = ResampleToUVTexture(img.shape[:2], base_order,
                                           sample_order, 1)

# Resample the image to textures (out: 3 x num_faces x H x W)
tex_image = resample_to_uv_layer(
    torch.from_numpy(img).float().permute(
        2, 0, 1).contiguous().unsqueeze(0)).squeeze(0).byte()

# ----------------------------------------------
# Compute SIFT descriptors for each patch
# ----------------------------------------------
kp_list = []  # Stores keypoint coords
desc_list = []  # Stores keypoint descriptors
quad_idx_list = []  # Stores quad index for each keypoint
for i in range(num_faces):
    kp_details = compute_sift_keypoints(tex_image[:, i, ...].permute(1, 2,
                                                                     0).numpy())

    if kp_details is not None:
        kp = kp_details[0]
        desc = kp_details[1]
        kp_list.append(kp)
        desc_list.append(desc)
        quad_idx_list.append(i * torch.ones(kp.shape[0]))

# Assemble keypoint data
patch_kp = torch.cat(kp_list, 0).float()  # M x 4 (x, y, s, o)
patch_desc = torch.cat(desc_list, 0).float()  # M x 128
patch_quad_idx = torch.cat(quad_idx_list, 0).long()  # M
print('Num Patch Keypoints:', patch_kp.shape[0])

if save_ply:
    # Convert the quad coordinates to normalized UV coords
    kp_uv = convert_quad_coord_to_uv((num_samples, num_samples),
                                     patch_kp[:, :2])

    # Convert the normalized quad UV data to 3D points
    kp_3d = convert_quad_uv_to_3d(patch_quad_idx, kp_uv, corners)

    # Create and save textured sphere
    tmp_img = torch.from_numpy(np.flip(img, 0).copy()).permute(
        2, 0, 1).float().unsqueeze(0)
    unresampler = Unresample('bispherical')
    sample_map = image_to_sphere_resample_map(tmp_img.shape[-2:], icosphere)
    rgb_vertices = unresampler(tmp_img.flip(-2), sample_map).squeeze()
    icosphere.normalize_points()
    write_ply(
        'textured_sphere.ply',
        icosphere.get_vertices().transpose(0, 1).numpy(),
        rgb=rgb_vertices.cpu().numpy(),
        faces=icosphere.get_all_face_vertex_indices().numpy(),
        text=False)

    # Write 3D keypoints
    write_ply('kp_patch_method.ply', kp_3d.view(-1, 3).numpy().T)

# ----------------------------------------------
# Compute only visible keypoints
# ----------------------------------------------
visible_kp, visible_desc = render_keypoints(img.shape[:2], patch_quad_idx,
                                            patch_kp, patch_desc, corners,
                                            (num_samples, num_samples))
print('patch')
print(patch_kp[4174, ...])
print(patch_desc[4174, ...])
print('returned')
print(visible_kp[-1, ...])
print(visible_desc[-1, ...])
print('Num Visible Keypoints:', visible_kp.shape[0])
if save_ply:
    kp_3d = convert_spherical_to_3d(
        convert_image_to_spherical(visible_kp[:, :2], img.shape[:2]))
    write_ply('kp_visible.ply', kp_3d.view(-1, 3).numpy().T)

if write_openmvg_files:
    np.savetxt(
        'patch_kp.feat',
        patch_kp.numpy().astype(np.float32),
        delimiter=' ',
        fmt='%f')
    np.savetxt(
        'patch_desc.desc',
        patch_desc.numpy().astype(np.uint32),
        delimiter=' ',
        fmt='%d')

# ----------------------------------------------
# Compute SIFT descriptors on equirect image
# ----------------------------------------------
erp_kp_details = compute_sift_keypoints(img)
erp_kp = erp_kp_details[0]
erp_desc = erp_kp_details[1]
print('Num Equirect. Keypoints:', erp_kp.shape[0])
if save_ply:
    kp_3d = convert_spherical_to_3d(
        convert_image_to_spherical(erp_kp[:, :2], img.shape[:2]))
    write_ply('kp_equirect_method.ply', kp_3d.view(-1, 3).numpy().T)

if write_openmvg_files:
    np.savetxt(
        'erp_kp.feat',
        erp_kp.numpy().astype(np.float32),
        delimiter=' ',
        fmt='%f')
    np.savetxt(
        'erp_desc.desc',
        erp_desc.numpy().astype(np.uint32),
        delimiter=' ',
        fmt='%d')

# ----------------------------------------------
# Plot descriptors
# ----------------------------------------------
# Create a figure and axis handle
fig, ax = plt.subplots(1, 1)

# Convert the normalized quad UV data to 3D points
patch_kp_on_img = convert_spherical_to_image(
    convert_3d_to_spherical(
        convert_quad_uv_to_3d(patch_quad_idx,
                              convert_quad_coord_to_uv(
                                  (num_samples, num_samples), patch_kp[:, :2]),
                              corners)), img.shape[:2])

# Set up the plot
ax.set_aspect(1, adjustable='box')
ax.imshow(img)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.plot(erp_kp[:, 0], erp_kp[:, 1], 'r.', markersize=3.0)
ax.plot(patch_kp_on_img[:, 0], patch_kp_on_img[:, 1], 'c.', markersize=3.0)
ax.plot(visible_kp[:, 0], visible_kp[:, 1], 'b.', markersize=3.0)
plt.show()