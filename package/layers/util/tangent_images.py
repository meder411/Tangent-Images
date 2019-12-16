import torch.nn as nn
import torch.nn.functional as F

import math

from tangent_images.nn import Unresample, Resample, ResampleFromUV
from .conversions import *
from .grids import *
import _tangent_images_ext._mesh as mesh

# -----------------------------------------------------------------------------


def generate_icosphere(order=0):
    '''
    Thanks to PyMesh for providing the initial icosahedron linkage
    https://github.com/PyMesh/PyMesh/blob/b5dafc28a9d2771b3e5294a17d1740cbdde7542f/python/pymesh/meshutils/generate_icosphere.py
    '''
    return mesh.generate_icosphere(order)


# -----------------------------------------------------------------------------


def compute_num_vertices(order):
    '''Computes the number of vertices for a given icosphere order'''
    v = 12 * (4**order)
    for i in range(order):
        v -= 6 * (4**i)
    return v


# -----------------------------------------------------------------------------


def compute_num_faces(order):
    '''Computes the number of vertices for a given icosphere order'''
    return 20 * (4**order)


# -----------------------------------------------------------------------------


def compute_num_samples(base_order, sample_order, kernel_size=None):
    '''Computes the number of samples for a tangent image dimension. If kernel size is provided, it returns the tangent image dimension with padding equal to floor(0.5 * kernel_size)'''
    num_samples = 2**(sample_order - base_order)
    if kernel_size is not None:
        num_samples += 2 * (kernel_size // 2)
    return num_samples


# -----------------------------------------------------------------------------


def get_sampling_resolution(base_order):
    '''
    After level 4, the vertex resolution comes pretty close to exactly halving at each subsequent order. This means we don't need to generate the sphere to compute the resolution. However, at lower levels of subdivision, we ought to compute the vertex resolution as it's not fixed.
    '''
    if base_order < 5:
        sampling_resolution = generate_icosphere(max(
            0, base_order - 1)).get_angular_resolution()
        if base_order == 0:
            sampling_resolution *= 2
    else:
        sampling_resolution = generate_icosphere(4).get_angular_resolution()
        sampling_resolution /= (2**(base_order - 5))
    return sampling_resolution


# -----------------------------------------------------------------------------


def gnomonic_kernel(spherical_coords, kh, kw, res_lat, res_lon):
    '''
    Creates gnomonic filters of shape (kh, kw) with spatial resolutions given by (res_lon, res_lat) and centers them at each coordinate given by <spherical_coords>

    spherical_coords: H, W, 2 (lon, lat)
    kh: vertical dimension of filter
    kw: horizontal dimension of filter
    res_lat: vertical spatial resolution of filter
    res_lon: horizontal spatial resolution of filter
    '''

    lon = spherical_coords[..., 0]
    lat = spherical_coords[..., 1]
    num_samples = spherical_coords.shape[0]

    # Kernel
    x = torch.zeros(kh * kw)
    y = torch.zeros(kh * kw)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)
            # Project the sphere onto the tangent plane
            x[i * kw + j] = cur_j * res_lon
            y[i * kw + j] = cur_i * res_lat

    # Center the kernel if dimensions are even
    if kh % 2 == 0:
        y += res_lat / 2
    if kw % 2 == 0:
        x += res_lon / 2

    # Equalize views
    lat = lat.view(1, num_samples, 1)
    lon = lon.view(1, num_samples, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    rho = (x**2 + y**2).sqrt()
    nu = rho.atan()
    out_lat = (nu.cos() * lat.sin() + y * nu.sin() * lat.cos() / rho).asin()
    out_lon = lon + torch.atan2(
        x * nu.sin(),
        rho * lat.cos() * nu.cos() - y * lat.sin() * nu.sin())

    # If kernel has an odd-valued dimension, handle the 0 case which resolves to NaN above
    if kh % 2 == 1:
        out_lat[..., [(kh // 2) * kw + kw // 2]] = lat
    if kw % 2 == 1:
        out_lon[..., [(kh // 2) * kw + kw // 2]] = lon

    # Compensate for longitudinal wrap around
    out_lon = ((out_lon + math.pi) % (2 * math.pi)) - math.pi

    # Return (1, num_samples, kh*kw, 2) map at locations given by <spherical_coords>
    return torch.stack((out_lon, out_lat), -1)


# -----------------------------------------------------------------------------


def gnomonic_kernel_from_sphere(icosphere,
                                kh,
                                kw,
                                res_lat,
                                res_lon,
                                source='vertex'):
    '''
    Returns a map of gnomonic filters with shape (kh, kw) and spatial resolution (res_lon, res_lat) centered at each vertex (or face) of the provided icosphere. Sample locations are given by spherical coordinates

    icosphere: icosphere object
    Kh: scalar height of planar kernel
    Kw: scalar width of planar kernel
    res_lat: scalar latitude resolution of kernel
    res_lon: scalar longitude resolution of kernel
    source: {'face' or 'vertex'}

    returns 1 x {F,V} x kh*kw x 2 sampling map per mesh element in spherical coords
    '''

    # Get lat/lon centers of convolution
    if source == 'face':
        spherical_coords = convert_3d_to_spherical(
            icosphere.get_face_barycenters())
        num_samples = icosphere.num_faces()
    elif source == 'vertex':
        spherical_coords = convert_3d_to_spherical(icosphere.get_vertices())
        num_samples = icosphere.num_vertices()
    else:
        print('Invalid source ({})'.format(source))
        exit()

    return gnomonic_kernel(spherical_coords, kh, kw, res_lat, res_lon)


# -----------------------------------------------------------------------------


def tangent_image_corners(icosphere, kh, kw, res_lat, res_lon, source='vertex'):
    '''
    Returns a map of gnomonic filters with shape (kh, kw) and spatial resolution (res_lon, res_lat) centered at each vertex (or face) of the provided icosphere. Sample locations are given by spherical coordinates

    icosphere: icosphere object
    Kh: scalar height of planar kernel
    Kw: scalar width of planar kernel
    res_lat: scalar latitude resolution of kernel
    res_lon: scalar longitude resolution of kernel
    source: {'face' or 'vertex'}

    returns 1 x {F,V} x kh*kw x 2 sampling map per mesh element in spherical coords
    '''

    # Get lat/lon centers of convolution
    if source == 'face':
        spherical_coords = convert_3d_to_spherical(
            icosphere.get_face_barycenters())
        num_samples = icosphere.num_faces()
    elif source == 'vertex':
        spherical_coords = convert_3d_to_spherical(icosphere.get_vertices())
        num_samples = icosphere.num_vertices()
    else:
        print('Invalid source ({})'.format(source))
        exit()

    # The "minus 1" term means that the center of the TL pixel is (0,0)
    return gnomonic_kernel(spherical_coords, 2, 2, (kh - 1) * res_lat,
                           (kw - 1) * res_lon)


# -----------------------------------------------------------------------------


def compute_tangent_image_angular_resolution(corners):
    '''
    corners: num_tangent_images x 4 x 3 (3d points)
    '''
    A = F.normalize(corners[..., 0, :], dim=-1)
    B = F.normalize(corners[..., 1, :], dim=-1)
    C = F.normalize(corners[..., 2, :], dim=-1)
    D = F.normalize(corners[..., 3, :], dim=-1)
    fov_x = (torch.acos((A * B).sum(-1)) * 180 / math.pi).mean()
    fov_y = (torch.acos((A * C).sum(-1)) * 180 / math.pi).mean()

    return fov_x, fov_y


# -----------------------------------------------------------------------------


def tangent_images_spherical_sample_map(base_order,
                                        sample_order,
                                        kernel_size=None):

    assert sample_order >= base_order, 'Sample order must be greater than or equal to the base order ({} <{ })'.format(
        sample_order, base_order)

    # Generate the base icosphere
    base_sphere = mesh.generate_icosphere(base_order)

    # Get sampling resolution
    sampling_resolution = get_sampling_resolution(base_order)

    # Determine how many samples to grab in each direction. Kernel size is used for grabbing padding.
    num_samples = compute_num_samples(base_order, sample_order)
    num_samples_with_pad = compute_num_samples(base_order, sample_order,
                                               kernel_size)

    # Generate spherical sample map s.t. each face is projected onto a tangent grid of size (num_samples x num_samples) and the samples are spaced (sampling_resolution/num_samples x sampling_resolution/num_samples apart)
    spherical_sample_map = gnomonic_kernel_from_sphere(
        base_sphere,
        num_samples_with_pad,
        num_samples_with_pad,
        sampling_resolution / num_samples,
        sampling_resolution / num_samples,
        source='face')

    return spherical_sample_map


# -----------------------------------------------------------------------------


def equirectangular_to_tangent_images_resample_map(image_shape,
                                                   base_order,
                                                   sample_order,
                                                   kernel_size=None):

    assert sample_order >= base_order, 'Sample order must be greater than or equal to the base order ({} <{ })'.format(
        sample_order, base_order)

    # Generate the spherical sample map
    spherical_sample_map = tangent_images_spherical_sample_map(
        base_order, sample_order, kernel_size)

    # Produces a sample map to turn the image into tangent planes
    image_sample_map = convert_spherical_to_image(spherical_sample_map,
                                                  image_shape)

    # Returns F_base x num_samples^2 x 2 sample map
    return image_sample_map.squeeze(0)


# -----------------------------------------------------------------------------


class ResampleToUVTexture(nn.Module):
    '''
    A class that maps a B x C x H x W image to B x F x C x P x P texture patches. These can be though of as a stack of F C x P x P patches. The memory layout is B x F x C x P x P in order to leverage grouped convolutions (groups = F).
    '''

    def __init__(
            self,
            image_shape,
            base_order,
            sample_order,
            kernel_size=1,  # Only not 1 when we want to use padding
            interpolation='bispherical'):

        super(ResampleToUVTexture, self).__init__()

        # Sample map
        self.register_buffer('sample_map',
                             equirectangular_to_tangent_images_resample_map(
                                 image_shape, base_order, sample_order,
                                 kernel_size))

        # Resampling layer
        self.layer = Unresample(interpolation)

        # Dimension of square tangent grid
        self.grid_dim = compute_num_samples(base_order, sample_order,
                                            kernel_size)

    def forward(self, x):
        '''
        Resample the image to the tangent planes as (B x C x F_base x num_samples^2)

        x: B x C x H x W

        returns B x C x F_base x H x W
        '''
        # Resample the image to the tangent planes as (B x F_base x C x num_samples^2)
        planes = self.layer(x, self.sample_map)

        # Reshape to a separate each patch
        B, C, N = planes.shape[:3]
        return planes.view(B, C, N, self.grid_dim, self.grid_dim)


# -----------------------------------------------------------------------------


class ResampleFromUVTexture(nn.Module):
    '''
    A class that maps B x F x C x P x P texture patches back to B x C x H x W image
    '''

    def __init__(self,
                 image_shape,
                 base_order,
                 sample_order,
                 interpolation='bilinear'):

        super(ResampleFromUVTexture, self).__init__()

        # Create base icosphere
        icosphere = mesh.generate_icosphere(base_order)

        # Compute number of samples and sampling resolution based on base and sample orders
        num_samples = compute_num_samples(base_order, sample_order)
        sampling_resolution = get_sampling_resolution(base_order)

        # Find the boundaries of the tangest planes in 3D
        corners = tangent_image_corners(icosphere, num_samples, num_samples,
                                        sampling_resolution / num_samples,
                                        sampling_resolution / num_samples,
                                        'face')

        corners = convert_spherical_to_3d(corners).squeeze()

        # Compute the rays for each pixel in the equirectangular image
        lat, lon = equirectangular_meshgrid(image_shape)[:2]
        rays = torch.stack((lon, lat), -1).view(-1, 2)

        quad, uv = mesh.find_tangent_plane_intersections(corners, rays)

        # Register the quad indices and UV coords as a buffer to bind it to module
        # Reshape quad and uv back to image dims
        # Scale normalized UV coords to actual (floating point) pixel coords
        self.register_buffer('quad', quad.view(*image_shape))
        self.register_buffer('uv', uv.view(*image_shape, 2) * (num_samples - 1))

        # Store number of faces
        self.F = icosphere.num_faces()

        # Create resampling layer
        self.layer = ResampleFromUV(interpolation)

    def forward(self, x):
        '''
        x: B x C x F x N x N tensor
        '''
        # Expand the tensor to a separate patches
        B = x.shape[0]
        N = x.shape[-1]
        x = x.view(B, -1, self.F, N, N)

        # Resample the image to the tangent planes as (B x C x OH x OW)
        return self.layer(x, self.quad, self.uv)


# -----------------------------------------------------------------------------


def get_tangent_image_info(base_order, sample_order, img_shape):

    # Patch size
    num_samples = compute_num_samples(base_order, sample_order)

    # ----------------------------------------------
    # Compute necessary data
    # ----------------------------------------------
    # Generate the base icosphere
    icosphere = mesh.generate_icosphere(base_order)

    # Number patches
    num_faces = compute_num_faces(base_order)

    # Sampling resolution
    sampling_resolution = get_sampling_resolution(base_order)

    # Corners of tangent planes in 3D coordinates
    corners = tangent_image_corners(icosphere, num_samples, num_samples,
                                    sampling_resolution / num_samples,
                                    sampling_resolution / num_samples, 'face')
    corners = convert_spherical_to_3d(corners).squeeze()

    # Module to resample the image to UV textures
    resample_to_uv_layer = ResampleToUVTexture(img_shape, base_order,
                                               sample_order, 1)

    return resample_to_uv_layer, corners