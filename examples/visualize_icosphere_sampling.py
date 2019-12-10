import torch
import torch.nn.functional as F
from tangent_images.util import *
import _tangent_images_ext._mesh as mesh

import math

kernel_size = 1  # Kernel size (so we know how much to pad)
base_order = 3  # Base sphere resolution
sample_order = 8  # Determines sample resolution
skip = 1  # How many to skip in sampling ones to visualize
show_tangent_with_padding = True  # Include padding samples in tangent plane visualizations

# Number of samples in each dimension
# 2^(difference between sample order and base sphere order) + padding
num_samples = 2**(sample_order - base_order)
num_samples_with_pad = num_samples + 2 * (kernel_size // 2)

# Generate the base icosphere
icosphere = mesh.generate_icosphere(base_order)

# After level 4, the vertex resolution comes pretty close to exactly halving at each subsequent order. This means we don't need to generate the sphere to compute the resolution. However, at lower levels of subdivision, we ought to compute the vertex resolution as it's not fixed
if base_order < 5:
    sampling_resolution = mesh.generate_icosphere(max(
        0, base_order - 1)).get_angular_resolution()
    if base_order == 0:
        sampling_resolution *= 2
else:
    sampling_resolution = mesh.generate_icosphere(5 -
                                                  1).get_angular_resolution()
    sampling_resolution /= (2**(base_order - 5))

# Generate the samples
spherical_sample_map = gnomonic_kernel_from_sphere(
    icosphere,
    num_samples_with_pad,
    num_samples_with_pad,
    sampling_resolution / num_samples,
    sampling_resolution / num_samples,
    source='face')
samples_3d = convert_spherical_to_3d(spherical_sample_map.squeeze())
samples_3d /= samples_3d.norm(dim=-1, keepdim=True)
samples_3d = samples_3d[::skip].contiguous()

# Corners of tangent planes
corners = tangent_plane_corners(
    icosphere,
    num_samples if not show_tangent_with_padding else num_samples_with_pad,
    num_samples if not show_tangent_with_padding else num_samples_with_pad,
    sampling_resolution / num_samples, sampling_resolution / num_samples,
    'face')
corners = convert_spherical_to_3d(corners).squeeze()
corners = corners[::skip, :].contiguous()

# Compute the angular resolution of the patches
A = F.normalize(corners[..., 0, :], dim=-1)
B = F.normalize(corners[..., 1, :], dim=-1)
C = F.normalize(corners[..., 2, :], dim=-1)
D = F.normalize(corners[..., 3, :], dim=-1)
fov_x = (torch.acos((A * B).sum(-1)) * 180 / math.pi).mean()
fov_y = (torch.acos((A * C).sum(-1)) * 180 / math.pi).mean()
print('Mean FOV X:', fov_x)
print('Mean FOV Y:', fov_y)
print('Mean Pixel Resolution X:', fov_x / num_samples)
print('Mean Pixel Resolution Y:', fov_y / num_samples)

# All vertex coords
colors = torch.zeros_like(samples_3d).byte()
face_vertex_colors = torch.zeros_like(corners).byte()
for i in range(colors.shape[0]):
    color = (torch.rand(1, 3) * 255).byte()
    colors[i, ...] = color
    face_vertex_colors[i, ...] = color

image_shape = (1000, 2000)
lat, lon = equirectangular_meshgrid(image_shape)[:2]
rays = convert_spherical_to_3d(torch.stack((lon, lat), -1).view(-1, 2))

# Write rays to pointclound
write_ply('rays.ply', rays.view(-1, 3).numpy().T)

# Write weighted vertices to pointclound
write_ply('weighted_vertices.ply',
          samples_3d.view(-1, 3).numpy().T,
          rgb=colors.view(-1, 3).numpy().T)

# Write icosphere to mesh
icosphere.normalize_points()
write_ply('icosphere.ply',
          icosphere.get_vertices().numpy().T,
          faces=icosphere.get_all_face_vertex_indices().numpy())

# Write tangent planes to mesh
faces = torch.zeros(2 * corners.shape[0], 3)
for i in range(corners.shape[0]):
    faces[2 * i, :] = torch.tensor([4 * i + 1, 4 * i + 3, 4 * i + 2])
    faces[2 * i + 1, :] = torch.tensor([4 * i, 4 * i + 1, 4 * i + 2])
write_ply('tangent_planes.ply',
          corners.view(-1, 3).numpy().T,
          rgb=face_vertex_colors.view(-1, 3).numpy().T,
          faces=faces.numpy())

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