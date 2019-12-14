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