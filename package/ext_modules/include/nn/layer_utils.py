import torch
from collections.abc import Iterable


def _ntuple(n):
    '''Function for handling scalar and iterable layer arguments'''

    def parse(x):
        '''Closure for parsing layer args'''
        if isinstance(x, Iterable):
            return x
        return tuple([x for i in range(n)])

    return parse


# Typedef
_pair = _ntuple(2)


def check_type(input, sample_map, interp_weights=None):

    assert input.dtype == sample_map.dtype, \
        'Input and sample map must be the same dtype ({} != {})'.format(
            input.dtype, sample_map.dtype)

    if interp_weights is not None:
        assert input.dtype == interp_weights.dtype, \
            'Input and interpolation weights must be the same dtype ' \
            '({} != {})'.format(input.dtype, interp_weights.dtype)


def check_input_dim(input):
    '''
    B, C, H, W
    '''
    assert input.dim() == 4, \
        'Input must have 4 dimensions ({} != 4)'.format(input.dim())


def check_sample_map_dim(sample_map, interp_weights=None, kernel=True):
    '''
    No interp weights:
            if kernel==True: OH, OW, K, 2
            else: OH, OW, 2
    With interp weights:
            if kernel==True: OH, OW, K, P, 2
            else: OH, OW, P, 2
    '''
    if interp_weights is None:
        if kernel:
            assert sample_map.dim() == 4, \
                'Sample map must have 4 dimensions ({} != 4)'.format(
                sample_map.dim())
        else:
            assert sample_map.dim() == 3, \
                'Sample map must have 3 dimensions ({} != 3)'.format(
                sample_map.dim())
    else:
        if kernel:
            assert sample_map.dim() == 5, \
                'Sample map must have 5 dimensions ({} != 5)'.format(
                sample_map.dim())
        else:
            assert sample_map.dim() == 4, \
                'Sample map must have 4 dimensions ({} != 4)'.format(
                sample_map.dim())

    assert sample_map.shape[-1] == 2, \
        'Last dimension of sample map must have size 2 ({} != 2)'.format(
        sample_map.shape[-1])


def check_interp_weights_dim(interp_weights, kernel=True):
    '''
    OH, OW, K, P
    '''
    if kernel:
        assert interp_weights.dim() == 4, \
            'Interpolation  weights must have 4 dimensions ({} != 4)'.format(
            interp_weights.dim())
    else:
        assert interp_weights.dim() == 3, \
            'Interpolation  weights must have 3 dimensions ({} != 3)'.format(
            interp_weights.dim())


def check_input_shape(input, in_channels):
    '''
    input: B, C, H, W
    '''
    assert input.shape[1] == in_channels, \
        'Number of input channels must match weight parameter ' \
        '({} != {})'.format(input.shape[1], in_channels)


def check_sample_map_shape(sample_map, kernel_size):
    '''
    sample_map: OH, OW, K, [P], 2
    '''
    assert sample_map.shape[2] == kernel_size, \
        'Kernel size of sample map must match weight parameter ' \
        '({} != {})'.format(sample_map.shape[2], kernel_size)


def check_interp_weights_shape(sample_map, interp_weights):
    '''
    sample_map: OH, OW, K, P, 2
    interp_weights: OH, OW, K, P
    '''
    assert sample_map.shape[-2] == interp_weights.shape[-1], \
        'Sample map and interpolation weights should have the same number' \
        'of interpolation points ({} != {})'.format(
        sample_map.shape[-2], interp_weights.shape[-1])


def check_input_map_shape(input, sample_map):
    assert input.shape[2] == sample_map.shape[0] and \
        input.shape[3] == sample_map.shape[1], \
        'Input dimensions 2 and 3 must match sample map dimensions 0 ' \
        'and 1 ({},{}) != ({},{}))'.format(input.shape[2], input.shape[3],
                                           sample_map.shape[0], sample_map.shape[1])


def check_args(input, sample_map, interp_weights, in_channels, kernel_size):

    # -----------------------------------
    # Check dimensions of arguments
    # -----------------------------------
    check_input_dim(input)
    check_sample_map_dim(sample_map, interp_weights, kernel_size is not None)
    if interp_weights is not None:
        check_interp_weights_dim(interp_weights, kernel_size is not None)

    # -----------------------------------
    # Check argument types
    # -----------------------------------
    check_type(input, sample_map, interp_weights)

    # -----------------------------------
    # Check inputs match network params
    # -----------------------------------
    if in_channels is not None:
        check_input_shape(input, in_channels)
    if kernel_size is not None:
        check_sample_map_shape(sample_map, kernel_size)
    if interp_weights is not None:
        check_interp_weights_shape(sample_map, interp_weights)
