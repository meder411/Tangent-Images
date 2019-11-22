__author__ = "Marc Eder"

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import torch
import torch.utils.data
import torch.nn.functional as F

import numpy as np
from skimage import transform, io

import math
import os
import os.path as osp
import json

from mapped_convolution.util import IntrinsicsModifier


class StanfordDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self,
                 root_path,
                 path_to_img_list,
                 fov=45,
                 dim=128,
                 scale_factor=1.0,
                 use_depth=False,
                 clip_depth=(0.0, 4.0),
                 data_format='data',
                 shift_x=15,   # 40
                 shift_y=15,   # 20
                 mean=None,
                 std=None):
        """
        fov (degrees) and dim determine the output dimension of the image
        Expects fov and dim to be tuples, e.g. (fx, fy)
        """

        # Set up a reader to load the panos
        self.root_path = root_path
        self.num_classes = 14

        # Create tuples of inputs/GT
        self.image_list = np.loadtxt(path_to_img_list, dtype=str)

        # Which type of data is this
        self.data_format = data_format

        # Other terms
        self.scale_factor = scale_factor
        self.fov = fov
        self.dim = dim
        self.shift_x = shift_x
        self.shift_y = shift_y

        # Note that this is the min FOV (in radians) of images the dataset. For consistency, we should not go above this because not all images will be able to match it
        self.max_fov = 0.7854028731372226
        if (math.radians(fov[0]) >= self.max_fov) or (math.radians(fov[1]) >=
                                                      self.max_fov):
            warn('Warning: Requested FOV is beyond maximum for dataset')

        if data_format == 'data':
            # Modules to resample the inputs
            self.bilinear_intrinsics_modifier = IntrinsicsModifier(
                *fov, *dim, 'bilinear')
            self.nearest_intrinsics_modifier = IntrinsicsModifier(
                *fov, *dim, 'nearest')

        self.use_depth = use_depth
        self.clip_depth = clip_depth

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        if use_depth:
            assert len(self.mean) == 4
        else:
            self.mean = self.mean[:3]
            self.std = self.std[:3]

    def __getitem__(self, idx, tocache=False):
        '''Load the inputs/GT at a given index'''

        # Select the pano set
        relative_paths = self.image_list[idx]

        # Load the pano set
        relative_basename = osp.splitext((relative_paths[0]))[0]
        basename = osp.splitext(osp.basename(relative_paths[0]))[0]

        cache_folder = osp.join(
            self.root_path, 'cache',
            'fold_1',
            'data_format_{}'.format(self.data_format),
            'scale_factor_{}'.format(self.scale_factor),
        )
        cache_path = osp.join(cache_folder, basename)
        if os.path.exists(cache_path):
            tmp = torch.load(cache_path)
            c = tmp.shape[0]
            rgb, labels = torch.split(tmp, [c-1, 1], dim=0)

            # Assemble the pano set
            pano_data = [rgb, labels, basename]

            # Return the set of pano data
            return pano_data

        rgb, mask = self.readRGBPano(osp.join(self.root_path,
                                              relative_paths[0]))
        labels = self.readSemanticPano(
            osp.join(self.root_path, relative_paths[2]))
        info = self.readPoseInfo(osp.join(self.root_path, relative_paths[3]))
        labels[mask == 0] = 0  # 14 previously

        # Convert to torch format from numpy -- i.e. make dimensions C x H x W
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().float()
        labels = torch.from_numpy(labels[None, ...]).float()

        if self.use_depth:
            depth_path = osp.join(self.root_path, relative_paths[1])
            depth = self.readDepthPano(depth_path)

            depth = torch.from_numpy(depth).to(torch.float32).unsqueeze(0)
            # TODO: should depth be resampled with nearest interpolation?
            rgb = torch.cat([rgb, depth], dim=0)

        # Center the data
        rgb -= torch.from_numpy(self.mean).view(-1, 1, 1)
        rgb /= torch.from_numpy(self.std).view(-1, 1, 1)

        if self.data_format == 'data':
            # Resample to normalize the camera matrices and FOV
            K = torch.tensor(info['camera_k_matrix'])
            # rgb = self.bilinear_intrinsics_modifier(rgb.unsqueeze(0),
            #                                         K).squeeze(0)
            # labels = self.nearest_intrinsics_modifier(labels.unsqueeze(0),
            #                                           K).squeeze(0)

            # Randomly shift the camera by up to +/-20 degrees in each direction to capture the whole image space
            out_shift_x = 2 * self.shift_x * torch.rand(1) - self.shift_x
            out_shift_y = 2 * self.shift_y * torch.rand(1) - self.shift_y
            rgb = self.bilinear_intrinsics_modifier(rgb.unsqueeze(0), K,
                                                    out_shift_x,
                                                    out_shift_y).squeeze(0)
            labels = self.nearest_intrinsics_modifier(labels.unsqueeze(0), K,
                                                      out_shift_x,
                                                      out_shift_y).squeeze(0)

        elif self.data_format == 'pano':
            # Resize the data
            rgb = F.interpolate(rgb.unsqueeze(0),
                                scale_factor=self.scale_factor,
                                mode='bilinear',
                                align_corners=False).squeeze(0)
            labels = F.interpolate(labels.unsqueeze(0),
                                   scale_factor=self.scale_factor,
                                   mode='nearest').squeeze(0)

        if tocache:
            if not osp.exists(cache_folder):
                os.makedirs(cache_folder, exist_ok=True)
            if os.path.exists(cache_path):
                raise IOError('{} already exists'.format(cache_path))
            tmp = torch.cat([rgb, labels], 0)
            torch.save(tmp, cache_path)
            return None

        # Assemble the pano set
        pano_data = [rgb, labels, basename]

        # Return the set of pano data
        return pano_data

    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readRGBPano(self, path):
        '''Reads the RGP pano image from file, normalized the RGB vaules to [0,1].'''

        # Normalize the image
        rgb = io.imread(path).astype(np.float32)[..., :3] / 255.
        mask = (rgb != 0).all(-1)

        return rgb, mask

    def readDepthPano(self, path):
        depth = io.imread(path)
        # missing values are encoded as 2^16 - 1
        missing_mask = (depth == 2 ** 16 - 1)
        # depth 0..128m is stretched to full uint16 range (1/512 m step)
        depth = depth.astype(np.float32) / 512.0
        # clip to a pre-defined range
        depth = np.clip(depth, self.clip_depth[0], self.clip_depth[1])
        # zero out missing values
        depth[missing_mask] = 0.0
        return depth

    def readSemanticPano(self, path):
        # Load the semantic labels
        return io.imread(path).astype(np.int32)

    def readPoseInfo(self, path):
        # Return the pose info as a dict
        with open(path, 'r') as f:
            info = json.load(f)
        return info
