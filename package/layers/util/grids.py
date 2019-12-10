import torch
import math


def get_equirectangular_grid_resolution(shape):
    '''Returns the resolution between adjacency grid indices for an equirectangular grid'''
    H, W = shape
    res_lat = math.pi / (H - 1)
    res_lon = 2 * math.pi / W
    return res_lat, res_lon


def equirectangular_meshgrid(shape):
    H, W = shape
    lat = torch.linspace(math.pi / 2, -math.pi / 2,
                         steps=H).view(-1, 1).expand(-1, W)
    lon = torch.linspace(-math.pi, math.pi,
                         steps=W + 1)[:-1].view(1, -1).expand(H, -1)
    res_lat, res_lon = get_equirectangular_grid_resolution(shape)
    return lat, lon, res_lat, res_lon


def cube_meshgrid_resolution(cube_dim):
    '''Returns the resolution between adjacency grid indices for cube map face grid'''
    return math.pi / (2 * cube_dim)


def cube_meshgrid(cube_dim):
    '''
    returns (v, u, index)
    '''
    H = cube_dim
    W = cube_dim * 6

    v = torch.arange(cube_dim).view(-1, 1).expand(-1, W)
    u = torch.arange(cube_dim).view(1, -1).expand(H, -1).repeat(1, 6)
    index = torch.zeros((H, W))
    for i in range(6):
        index[:, i * cube_dim:(i + 1) * cube_dim] += i

    return v, u, index


def cartesian_meshgrid(shape):
    '''Create a Cartesian meshgrid'''
    H, W = shape
    y = torch.arange(H).view(-1, 1).expand(-1, W).float()
    x = torch.arange(W).view(1, -1).expand(H, -1).float()
    return y, x


def normalized_meshgrid(shape):
    """
    shape: (H, W)
    Returns a meshgrid normalized such that all values are in [-1, 1]. Like with PyTorch.grid_sample: values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.
    """
    H, W = shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.float()
    grid_y = grid_y.float()

    # Precompute some useful values
    half_width = ((W - 1) / 2)
    half_height = ((H - 1) / 2)

    # Center the grid
    grid_x -= half_width
    grid_y -= half_height

    # Normalize the grid
    grid_x /= half_width
    grid_y /= half_height

    return grid_y, grid_x