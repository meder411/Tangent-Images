__author__ = "Marc Eder"

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import torch
import torch.utils.data
import torch.nn.functional as F

import numpy as np
from skimage import io

import math
import os
import os.path as osp
import json

from spherical_distortion.transforms import CameraNormalization
from spherical_distortion.util import InterpolationType, numpy2torch


class StanfordDataset(torch.utils.data.Dataset):
    """PyTorch dataset module for effiicient loading"""

    def __init__(
            self,
            root_path,
            path_to_img_list,
            fold,
            normalize_intrinsics=True,
            fov=(45, 45),
            dim=(128, 128),
            scale_factor=1.0,
            use_depth=False,
            clip_depth=(0.0, 4.0),
            data_format='data',
            cache_root=None,  # If desiring a different cache location than root
            mean=None,
            std=None):
        """
        :param root_path: path to the root data folder.
           Expected structure:
              area_1
              area_2
              ...
              area_6
              cache (optional)
        :param path_to_img_list: path to list of relative paths to rgb/depth/anno
        :param fov (degrees) and dim determine the output dimension of the image
           Expects fov and dim to be tuples, e.g. (fx, fy)
        :param dim (int)
        :param scale_factor
        :param use_depth (bool) use depth channel
        :param clip_depth (tuple, meters): clip depth to this range
        :param data_format (data or pano)
        :param mean (tuple) mean for normalization
        :param std (tuple) std for normalization
        """

        self.root_path = root_path
        self.num_classes = 14
        self.pano_shape = (2048, 4096)
        self.fold = fold

        # Create tuples of inputs/GT
        self.image_list = np.loadtxt(path_to_img_list, dtype=str)

        # Which type of data is this
        self.data_format = data_format

        # Other terms
        self.scale_factor = scale_factor
        self.fov = fov
        self.dim = dim

        # Note that this is the max FOV (in radians) of images the dataset.
        # For consistency, we should not go above this
        # because not all images will be able to match it.
        self.max_fov = 0.7854028731372226
        if (math.radians(fov[0]) >= self.max_fov
                or math.radians(fov[1]) >= self.max_fov):
            raise AttributeError(
                'Requested FOV is beyond maximum allowed for dataset')

        if data_format == 'data':
            # Modules to resample the inputs
            self.camnorm_bilinear = CameraNormalization(fov, dim, False)
            self.camnorm_nearest = CameraNormalization(
                fov, dim, False, InterpolationType.NEAREST)

        self.normalize_intrinsics = normalize_intrinsics
        self.use_depth = use_depth
        self.clip_depth = clip_depth

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        if use_depth:
            assert len(self.mean) == 4
        else:
            self.mean = self.mean[:3]
            self.std = self.std[:3]

        # Specify the caching folder
        self.cache_folder = osp.join(cache_root
                                     if cache_root else self.root_path,
                                     'cache', 'stanford', '{}'.format(
                                         self.data_format))
        if self.use_depth:
            self.cache_folder = osp.join(self.cache_folder, 'with_depth')
        if self.data_format == 'pano':
            self.cache_folder = osp.join(self.cache_folder,
                                         'scale_factor_{}'.format(
                                             self.scale_factor))

    def random_crops(self, rgb, labels, K):
        """
        Compute a random shift (could use either camnorm instance as both have same intrinsic parameters)
        """
        shift = self.camnorm_bilinear.compute_random_shift(rgb.shape[-2:], K)

        # Normalize with a random crop
        rgb = self.camnorm_bilinear(rgb, K, shift)
        labels = self.camnorm_nearest(labels, K, shift)

        return rgb, labels

    def __getitem__(self, idx, tocache=False):
        """Load the inputs/GT at a given index"""
        if tocache:
            assert self.data_format == 'pano', 'Caching only supported for pano mode'

        # Select the data
        relative_paths = self.image_list[idx]

        # Load the data
        basename = osp.splitext(osp.basename(relative_paths[0]))[0]
        cache_path = osp.join(self.cache_folder, basename)
        if os.path.exists(cache_path):
            if tocache:
                # If we're trying to cache the data, and it's already there, don't do anything
                return
            tmp = torch.load(cache_path)
            c = tmp.shape[0]
            rgb, labels = torch.split(tmp, [c - 1, 1], dim=0)

            # Assemble the pano set
            pano_data = [rgb, labels, basename]

            # Return the set of pano data
            return pano_data

        # Otherwise, continue on to data loading
        rgb, mask = self.read_rgb(osp.join(self.root_path, relative_paths[0]))
        labels = self.read_semantic(
            osp.join(self.root_path, relative_paths[2]))
        info = self.read_pose_info(osp.join(self.root_path, relative_paths[3]))
        labels[mask == 0] = 0

        # Convert to torch format from numpy -- i.e. make dimensions C x H x W
        rgb = numpy2torch(rgb).float()
        labels = numpy2torch(labels)[None, ...].float()

        if self.use_depth:
            depth_path = osp.join(self.root_path, relative_paths[1])
            depth = self.read_depth(depth_path)
            depth = numpy2torch(depth[..., None]).float()
            rgb = torch.cat([rgb, depth], dim=0)

        # Center the data
        rgb -= torch.from_numpy(self.mean).view(-1, 1, 1)
        rgb /= torch.from_numpy(self.std).view(-1, 1, 1)

        # Camera normalize the perspective images
        if self.data_format == 'data':
            if self.normalize_intrinsics:
                K = torch.tensor(info['camera_k_matrix'])
                rgb, labels = self.random_crops(rgb, labels, K)

        # Resize the pano images as desired
        elif self.data_format == 'pano':
            rgb = F.interpolate(
                rgb.unsqueeze(0),
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False,
                recompute_scale_factor=True).squeeze(0)
            labels = F.interpolate(
                labels.unsqueeze(0),
                scale_factor=self.scale_factor,
                mode='nearest',
                recompute_scale_factor=True).squeeze(0)

            # If caching data
            if tocache:
                if not osp.exists(self.cache_folder):
                    os.makedirs(self.cache_folder, exist_ok=True)
                if os.path.exists(cache_path):
                    raise IOError('{} already exists'.format(cache_path))
                tmp = torch.cat([rgb, labels], 0)
                torch.save(tmp, cache_path, pickle_protocol=4)

                return None

        # Assemble the pano set
        data = [rgb, labels, basename]

        # Return the set of pano data
        return data

    def __len__(self):
        """Return the size of this dataset"""
        return len(self.image_list)

    @staticmethod
    def read_rgb(path):
        """
        Reads the RGP pano image from file,
        normalized the RGB values to [0,1].
        """

        # Normalize the image
        rgb = io.imread(path).astype(np.float32)[..., :3] / 255.
        mask = (rgb != 0).all(-1)
        return rgb, mask

    def read_depth(self, path):
        depth = io.imread(path)
        # missing values are encoded as 2^16 - 1
        missing_mask = (depth == 2**16 - 1)
        # depth 0..128m is stretched to full uint16 range (1/512 m step)
        depth = depth.astype(np.float32) / 512.0
        # clip to a pre-defined range
        depth = np.clip(depth, self.clip_depth[0], self.clip_depth[1])
        # zero out missing values
        depth[missing_mask] = 0.0
        return depth

    @staticmethod
    def read_semantic(path):
        # Load the semantic labels
        return io.imread(path).astype(np.int32)

    @staticmethod
    def read_pose_info(path):
        # Return the pose info as a dict
        with open(path, 'r') as f:
            info = json.load(f)
        return info
