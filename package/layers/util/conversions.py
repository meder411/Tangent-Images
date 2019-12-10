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


def convert_3d_to_cube(xyz, cube_dim):
    '''
    xyz : * x 3
    returns : * x 2 (u,v), * x 1 (index)
    '''
    # For convenience
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    absX = x.abs()
    absY = y.abs()
    absZ = z.abs()
    isXPositive = x > 0
    isYPositive = y > 0
    isZPositive = z > 0

    uv = torch.zeros(*xyz.shape[:-1], 2).float()
    index = torch.zeros(*xyz.shape[:-1], 1).long()

    # POSITIVE X
    # u (0 to 1) goes from +z to -z
    # v (0 to 1) goes from +y to -y
    cube = isXPositive & (absX >= absY) & (absX >= absZ)
    uv[..., 0][cube] = 0.5 * (-z[cube] / absX[cube] + 1)
    uv[..., 1][cube] = 0.5 * (-y[cube] / absX[cube] + 1)
    index[..., 0][cube] = 3

    # NEGATIVE X
    # u (0 to 1) goes from -z to +z
    # v (0 to 1) goes from +y to -y
    cube = ~isXPositive & (absX >= absY) & (absX >= absZ)
    uv[..., 0][cube] = 0.5 * (z[cube] / absX[cube] + 1)
    uv[..., 1][cube] = 0.5 * (-y[cube] / absX[cube] + 1)
    index[..., 0][cube] = 1

    # POSITIVE Y
    # u (0 to 1) goes from -x to +x
    # v (0 to 1) goes from -z to +z
    cube = isYPositive & (absY >= absX) & (absY >= absZ)
    uv[..., 0][cube] = 0.5 * (x[cube] / absY[cube] + 1)
    uv[..., 1][cube] = 0.5 * (z[cube] / absY[cube] + 1)
    index[..., 0][cube] = 4

    # NEGATIVE Y
    # u (0 to 1) goes from -x to +x
    # v (0 to 1) goes from +z to -z
    cube = ~isYPositive & (absY >= absX) & (absY >= absZ)
    uv[..., 0][cube] = 0.5 * (x[cube] / absY[cube] + 1)
    uv[..., 1][cube] = 0.5 * (-z[cube] / absY[cube] + 1)
    index[..., 0][cube] = 5

    # POSITIVE Z
    # u (0 to 1) goes from -x to +x
    # v (0 to 1) goes from +y to -y
    cube = isZPositive & (absZ >= absX) & (absZ >= absY)
    uv[..., 0][cube] = 0.5 * (x[cube] / absZ[cube] + 1)
    uv[..., 1][cube] = 0.5 * (-y[cube] / absZ[cube] + 1)
    index[..., 0][cube] = 2

    # NEGATIVE Z
    # u (0 to 1) goes from +x to -x
    # v (0 to 1) goes from +y to -y
    cube = ~isZPositive & (absZ >= absX) & (absZ >= absY)
    uv[..., 0][cube] = 0.5 * (-x[cube] / absZ[cube] + 1)
    uv[..., 1][cube] = 0.5 * (-y[cube] / absZ[cube] + 1)
    index[..., 0][cube] = 0

    # Convert all [0,1] coords to cube coord
    uv *= cube_dim

    return uv, index


def convert_spherical_to_cube(lonlat, cube_dim):
    '''
    Convenience function
    '''
    return convert_3d_to_cube(convert_spherical_to_3d(lonlat), cube_dim)


def convert_cube_to_3d(uv, index, cube_dim):
    '''
    Indexing is to [-z, -x, +z, +x, +y, -y]
    Assumes that pixel centers are (u + 0.5, v + 0.5)
    uv : * x 2
    index : * x 1
    returns : * x 3 (xyz)
    '''
    # Convert from cube coord to [0,1]
    uv = uv.float()
    uv = (uv + 0.5) / cube_dim

    # Convert from [0,1] range to [-1,1]
    uv = 2 * uv - 1

    # For convenience
    u = uv[..., 0]
    v = uv[..., 1]

    xyz = torch.zeros(*uv.shape[:-1], 3).float()

    # POSITIVE X
    case_3 = index == 3
    xyz[..., 0][case_3] = 1.0
    xyz[..., 1][case_3] = -v[case_3]
    xyz[..., 2][case_3] = -u[case_3]

    # NEGATIVE X
    case_1 = index == 1
    xyz[..., 0][case_1] = -1.0
    xyz[..., 1][case_1] = -v[case_1]
    xyz[..., 2][case_1] = u[case_1]

    # POSITIVE Y
    case_4 = index == 4
    xyz[..., 0][case_4] = u[case_4]
    xyz[..., 1][case_4] = 1.0
    xyz[..., 2][case_4] = v[case_4]

    # NEGATIVE Y
    case_5 = index == 5
    xyz[..., 0][case_5] = u[case_5]
    xyz[..., 1][case_5] = -1.0
    xyz[..., 2][case_5] = -v[case_5]

    # POSITIVE Z
    case_2 = index == 2
    xyz[..., 0][case_2] = u[case_2]
    xyz[..., 1][case_2] = -v[case_2]
    xyz[..., 2][case_2] = 1.0

    # NEGATIVE Z
    case_0 = index == 0
    xyz[..., 0][case_0] = -u[case_0]
    xyz[..., 1][case_0] = -v[case_0]
    xyz[..., 2][case_0] = -1.0

    return xyz


def convert_cubemap_tuple_to_image(uv, idx, cube_dim):
    '''Converts cubemap coordinates from tuple form (uv, cube_idx) to coordinates on the (cube_dim x 6*cube_dim) cubemap image'''
    u = uv[..., 0]
    v = uv[..., 1]
    idx = idx[..., 0]
    y = v
    x = idx.float() * cube_dim + u
    return torch.stack((x, y), -1)


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