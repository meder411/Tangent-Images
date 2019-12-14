import torch

import math


def convert_spherical_to_image(rad, shape):
    '''
    rad: * x ... x * x 2 (lon, lat)

    returns: * x ... x * x 2 (x,y)
    '''
    H, W = shape
    xy = torch.zeros_like(rad)
    xy[..., 0] = (W / 2) * (rad[..., 0] / math.pi + 1)
    xy[..., 1] = ((H - 1) / 2) * (-2 * rad[..., 1] / math.pi + 1)
    return xy


def convert_image_to_spherical(xy, shape):
    '''
    xy: * x ... x * x 2 (x, y)

    returns: * x ... x * x 2 (lon, lat)
    '''
    H, W = shape
    rad = torch.zeros_like(xy)
    rad[..., 0] = math.pi * (2 * xy[..., 0] / W - 1)
    rad[..., 1] = (-math.pi / 2) * (2 * xy[..., 1] / (H - 1) - 1)
    return rad


def convert_3d_to_spherical(xyz):
    '''
    xyz : * x 3
    returns : * x 2 (lon, lat)
    '''
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    lat = torch.atan2(-y, (x**2 + z**2).sqrt())
    lon = torch.atan2(x, z)
    return torch.stack((lon, lat), -1)


def convert_spherical_to_3d(lonlat):
    '''
    xyz : * x 2 (lon, lat)
    returns : * x 3 (x,y,z)
    '''
    x = lonlat[..., 1].cos() * lonlat[..., 0].sin()
    y = -lonlat[..., 1].sin()
    z = lonlat[..., 1].cos() * lonlat[..., 0].cos()
    return torch.stack((x, y, z), -1)


def convert_quad_coord_to_uv(quad_shape, coord):
    """
    quad_shape: (H, W)
    coord: N x 2 (X, Y)

    returns: N x 2 (u, v)
    """
    uv = torch.zeros_like(coord)
    H, W = quad_shape
    uv[:, 0] = coord[:, 0] / W
    uv[:, 1] = coord[:, 1] / H
    return uv


def convert_quad_uv_to_3d(quad_idx, uv, quad_corners):
    """
    quad_idx: M
    uv: M x 2
    quad_corners: N x 4 x 3

    returns: M x 3 3D points
    """
    # Grab the relevant quad data (out: M x 4 x 3)
    relevant_quads = quad_corners[quad_idx, ...]

    # Vectors defining the quads (each out: M x 3)
    u_vec = relevant_quads[:, 1, :] - relevant_quads[:, 0, :]
    v_vec = relevant_quads[:, 2, :] - relevant_quads[:, 0, :]

    # Convenience
    u = uv[:, [0]]
    v = uv[:, [1]]

    # Compute 3D point on the quad as vector addition
    pts_3d = relevant_quads[:, 0, :] + u * u_vec + v * v_vec

    return pts_3d