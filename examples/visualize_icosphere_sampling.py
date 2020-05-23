import torch
from spherical_distortion.util import *
import os

# ---------------
# Parameters
# ---------------
base_order = 1  # Base sphere resolution
sample_order = 6  # Determines sample resolution
skip = 10  # How many tangent images to skip in sampling ones to visualize

# ---------------
# Compute the necessary informations
# ---------------
# Create the sampling map from the icosahedron
spherical_sample_map = tangent_images_spherical_sample_map(
    base_order, sample_order)
samples_3d = convert_spherical_to_3d(spherical_sample_map.squeeze())
samples_3d /= samples_3d.norm(dim=-1, keepdim=True)
samples_3d = samples_3d[::skip].contiguous()

# Corners of tangent planes
corners = tangent_image_corners(base_order, sample_order)
corners = corners[::skip, :].contiguous()

# ---------------
# Random Aside...
# ---------------
# For fun, let's compute the angular resolution of the patches
dim = tangent_image_dim(base_order, sample_order)
fov_x, fov_y = compute_tangent_image_angular_resolution(corners)
print('Mean FOV X:', fov_x)
print('Mean FOV Y:', fov_y)
print('Mean Pixel Resolution X:', fov_x / dim)
print('Mean Pixel Resolution Y:', fov_y / dim)

# ---------------
# Visualization
# ---------------
os.makedirs('outputs', exist_ok=True)

# Create random color assignments for each tangent image
colors = torch.zeros_like(samples_3d).byte()
face_vertex_colors = torch.zeros_like(corners).byte()
for i in range(colors.shape[0]):
    color = (torch.rand(1, 3) * 255).byte()
    colors[i, ...] = color
    face_vertex_colors[i, ...] = color

# Write the sample locations *on the sphere* that correspond to each tangent image to a point cloud
write_ply(
    'outputs/samples_3d.ply',
    samples_3d.view(-1, 3).numpy().T,
    rgb=colors.view(-1, 3).numpy().T)

# Write base icosphere to mesh
icosphere = generate_icosphere(base_order)
icosphere.normalize_points()
write_ply(
    'outputs/icosphere.ply',
    icosphere.get_vertices().numpy().T,
    faces=icosphere.get_all_face_vertex_indices().T.numpy())

# Write tangent planes to mesh
faces = torch.zeros(2 * corners.shape[0], 3)
for i in range(corners.shape[0]):
    faces[2 * i, :] = torch.tensor([4 * i + 1, 4 * i + 3, 4 * i + 2])
    faces[2 * i + 1, :] = torch.tensor([4 * i, 4 * i + 1, 4 * i + 2])
write_ply(
    'outputs/tangent_planes.ply',
    corners.view(-1, 3).numpy().T,
    rgb=face_vertex_colors.view(-1, 3).numpy().T,
    faces=faces.T.numpy())