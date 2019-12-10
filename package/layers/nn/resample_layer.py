import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

import math
import time

import _tangent_images_ext._resample as resample
import _tangent_images_ext._weighted_resample as weighted_resample
import _tangent_images_ext._uv_resample as uv_resample

from .layer_utils import *


class ResampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                sample_map,
                output_shape,
                interp,
                interp_weights=None):
        self.save_for_backward(input, sample_map, torch.tensor(interp),
                               interp_weights)

        if interp_weights is not None:
            return weighted_resample.weighted_resample_to_map(
                input, sample_map, interp_weights, output_shape[0],
                output_shape[1], interp)
        else:
            return resample.resample_to_map(input, sample_map, output_shape[0],
                                            output_shape[1], interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            sample_map, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            unresampled_grad_output = weighted_resample.weighted_resample_from_map(
                grad_output, sample_map, interp_weights, interp)
        else:
            unresampled_grad_output = resample.resample_from_map(
                grad_output, sample_map, interp)

        return unresampled_grad_output, None, None, None, None


class Resample(nn.Module):
    '''
    A class that maps integer-valued input locations to real-valued output locations according to a function.
    '''

    def __init__(self, interpolation='bilinear'):

        super(Resample, self).__init__()

        if interpolation == 'nearest':
            self.interp = 0
        elif interpolation == 'bilinear':
            self.interp = 1
        elif interpolation == 'bispherical':
            self.interp = 2
        else:
            assert False, 'Unsupported interpolation type'

    def forward(self, x, sample_map, output_shape, interp_weights=None):
        '''
        x:              batch x channels x input_height x input_width
        sample_map: input_height x input_width x 2 (x, y)
        output_shape:   (output_height, output_width)
        interp_weights: [OPTIONAL] input_height x input_width x num_interp_points x 2 (x, y)
        '''
        check_args(x, sample_map, interp_weights, None, None)
        check_input_map_shape(x, sample_map)
        return ResampleFunction.apply(x, sample_map, output_shape, self.interp,
                                      interp_weights)


class UnresampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input, sample_map, interp, interp_weights=None):
        self.save_for_backward(input, sample_map, torch.tensor(interp),
                               interp_weights)

        if interp_weights is not None:
            return weighted_resample.weighted_resample_from_map(
                input, sample_map, interp_weights, interp)
        else:
            return resample.resample_from_map(input, sample_map, interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            sample_map, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            resampled_grad_output = weighted_resample.weighted_resample_to_map(
                grad_output, sample_map, interp_weights, input.shape[2],
                input.shape[3], interp)
        else:
            resampled_grad_output = resample.resample_to_map(
                grad_output, sample_map, input.shape[2], input.shape[3], interp)

        return resampled_grad_output, None, None, None, None


class Unresample(nn.Module):
    '''
    A class that maps real-valued input locations to a integer-valued output location according to a function. Essentially a mapped convolution with unit weights, no bias, and a kernel size of 1.
    '''

    def __init__(self, interpolation='bilinear'):

        super(Unresample, self).__init__()

        if interpolation == 'nearest':
            self.interp = 0
        elif interpolation == 'bilinear':
            self.interp = 1
        elif interpolation == 'bispherical':
            self.interp = 2
        else:
            assert False, 'Unsupported interpolation type'

    def forward(self, x, sample_map, interp_weights=None):
        '''
        x:              batch x channels x input_height x input_width
        sample_map:     output_height x output_width x 2 (x, y)
        interp_weights: output_height x output_width x num_interp_points x 2 (x, y)
        '''
        check_args(x, sample_map, interp_weights, None, None)

        return UnresampleFunction.apply(x, sample_map, self.interp,
                                        interp_weights)


class ResampleFromUVFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input, quad_idx, tex_uv, interp):
        self.save_for_backward(quad_idx, tex_uv, torch.tensor(interp),
                               torch.tensor(input.shape[-3:]))

        return uv_resample.resample_from_uv_maps(input, quad_idx, tex_uv,
                                                 interp)

    @staticmethod
    def backward(self, grad_output):
        quad_idx, \
        tex_uv, \
        interp, \
        input_shape = self.saved_tensors

        resampled_grad_output = uv_resample.resample_to_uv_maps(
            grad_output, quad_idx, tex_uv, input_shape[0], input_shape[1],
            input_shape[2], interp)

        return resampled_grad_output, None, None, None


class ResampleFromUV(nn.Module):
    '''
    A class that maps real-valued input locations on a B x C x N x H x W set of N HxW textures to a integer-valued output location.
    '''

    def __init__(self, interpolation='bilinear'):

        super(ResampleFromUV, self).__init__()

        if interpolation == 'nearest':
            self.interp = 0
        elif interpolation == 'bilinear':
            self.interp = 1
        else:
            assert False, 'Unsupported interpolation type'

    def forward(self, x, quad_idx, tex_uv):
        '''
        x:           batch x channels x num_textures x tex_height x tex_width
        quad_idx:    output_height x output_width
        tex_uv:      output_height x output_width x 2 (x, y)
        '''

        assert x.dim() == 5, \
            'Input expected to be 5 dimensional tensor ({} != {})'.format(
            x.dim(), 5)
        assert (quad_idx.dim() == 2) and (quad_idx.dtype == torch.int64), \
            'quad_idx expected to be 2 dimension tensor of type long'
        assert tex_uv.shape[:2] == quad_idx.shape, \
            'tex_uv expected to have same first two dimensions as quad_idx ({} != {})'.format(tex_uv.shape[:2], quad_idx.shape)

        return ResampleFromUVFunction.apply(x, quad_idx, tex_uv, self.interp)