import os
import glob
from torch.utils import data
from PIL import Image
import torchvision
import numpy as np
import torch
import os.path as osp
import cv2
import torch.nn.functional as F

from spherical_distortion.transforms import CameraNormalization
from spherical_distortion.util import InterpolationType, numpy2torch


class OmniSynth(data.Dataset):
    """
    OmniSynthia dataset. Equirectangular images for SYNTHIA sequences.
    """

    def __init__(self,
                 omni_dp='./data/omnisynthia',
                 is_train=True,
                 fov=(45, 45),
                 dim=(128, 128),
                 scale_factor=1.0,
                 data_format='data',
                 cache_root=None,
                 mean=None,
                 std=None):
        """Initialization"""
        self.root_path = omni_dp
        self.is_train = is_train
        self.data_format = data_format

        self.pano_shape = (2096, 4192)

        self.num_classes = 14
        self.ignore_label = 0

        self.scale_factor = scale_factor

        if self.is_train:
            extensions = [
                'SYNTHIA-SEQS-01-SUMMER', 'SYNTHIA-SEQS-02-SUMMER',
                'SYNTHIA-SEQS-05-SUMMER', 'SYNTHIA-SEQS-06-SUMMER'
            ]
        else:
            extensions = ['SYNTHIA-SEQS-04-SUMMER']

        self.image_list = []

        # Disambiguate directory names depending on file format
        if data_format == 'data':
            labels_folder = 'GT/LABELS'
            subst_index = -4
            template = os.path.join('RGB', 'Stereo_*', 'Omni_*', '*.png')
        else:
            labels_folder = 'GT/LABELS_Correct'
            subst_index = -2
            template = os.path.join('pano', 'RGB', '*.png')

        # Create the list of filepaths
        for ext in extensions:
            seq_template = os.path.join(self.root_path, ext, template)

            for full_im_path in glob.glob(seq_template):
                parts = full_im_path.split('/')
                parts[subst_index] = labels_folder
                parts = ['/'] + parts
                labels_path = os.path.join(*parts)

                self.image_list.append((full_im_path, labels_path))

        # Pre-processing module to convert numpy to torch format and normalize the input channels
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])

        # Specify the caching folder
        self.cache_folder = osp.join(cache_root
                                     if cache_root else self.root_path,
                                     'cache', 'omnisynthia', '{}'.format(
                                         self.data_format))
        if self.data_format == 'pano':
            self.cache_folder = osp.join(self.cache_folder,
                                         'scale_factor_{}'.format(
                                             self.scale_factor))

        # Initialize the camera normalization transform modules
        if data_format == 'data':
            # Intrinsics information for all SYNTHIA perspective images
            fx = fy = 532.7403520000000
            cx = 640
            cy = 380
            self.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.camnorm_bilinear = CameraNormalization(fov, dim, False)
            self.camnorm_nearest = CameraNormalization(
                fov, dim, False, InterpolationType.NEAREST)

    def random_crops(self, rgb, labels, K):
        """
        Compute a random shift (could use either camnorm instance as both have same intrinsic parameters)
        """
        shift = self.camnorm_bilinear.compute_random_shift(rgb.shape[-2:], K)

        # Normalize with a random crop
        rgb = self.camnorm_bilinear(rgb, K, shift)
        labels = self.camnorm_nearest(labels, K, shift)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index, tocache=False):
        if tocache and (self.data_format == 'data'):
            assert False, 'Caching only enabled for pano format'

        # Select the data
        rgb_img_fp, labels_color_fp = self.image_list[index]

        # Load the data
        basename = osp.splitext(osp.basename(rgb_img_fp))[0]
        cache_path = osp.join(self.cache_folder,
                              osp.splitext(
                                  osp.relpath(rgb_img_fp, self.root_path))[0])
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

        # Load the label files using OpenCV
        if self.data_format == 'data':
            # uint16 image, labels are stored in the last channel
            labels = cv2.imread(labels_color_fp, cv2.IMREAD_UNCHANGED)[:, :, 2]
        else:
            # uint8 image, value=255 introduced for invalid region
            labels = cv2.imread(labels_color_fp, cv2.IMREAD_UNCHANGED)
            labels[labels == 255] = 0
        labels = labels.astype('float32')

        # Handle missing classes
        unq = np.unique(labels.tolist())
        assert not 13 in unq and not 14 in unq
        labels[labels == 15] = 13

        # Load the RGB image using PIL
        rgb = Image.open(rgb_img_fp)

        # Preprocess the loaded data for training
        rgb = self.preprocess(rgb)
        labels = numpy2torch(labels).unsqueeze(0)

        # Camera normalize the perspective images
        if self.data_format == 'data':
            K = torch.tensor(self.K).view(3, 3)

            # Compute a random shift (could use either camnorm instance as both have same intrinsic parameters)
            shift = self.camnorm_bilinear.compute_random_shift(
                rgb.shape[-2:], K)

            # Normalize with a random crop
            rgb = self.camnorm_bilinear(rgb, K, shift)
            labels = self.camnorm_nearest(labels, K, shift)

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
                cache_dir = osp.dirname(cache_path)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                if os.path.exists(cache_path):
                    raise IOError('{} already exists'.format(cache_path))
                tmp = torch.cat([rgb, labels], 0)
                torch.save(tmp, cache_path, pickle_protocol=4)
                return None

        # Assemble the pano set
        pano_data = [rgb, labels, basename]

        # Return the set of pano data
        return pano_data


if __name__ == '__main__':
    # [1.70271852e-01 1.85291552e-01 2.07662872e-01 2.52132621e-01
    #  3.14578630e-02 3.89688762e-02 3.64869576e-02 9.81838729e-03
    #  4.16962315e-02 1.49937989e-03 1.48623227e-03 5.64423593e-06
    #  2.26893700e-02 5.32160971e-04]
    # w/o class=0
    # [2.2332e-01, 2.5028e-01, 3.0387e-01, 3.7913e-02, 4.6966e-02, 4.3975e-02,
    #  1.1833e-02, 5.0253e-02, 1.8071e-03, 1.7912e-03, 6.8025e-06, 2.7346e-02,
    #  6.4137e-04]
    import tqdm
    dataset = OmniSynth(data_format='data')
    num_classes = dataset.num_classes
    hist_all = np.zeros(num_classes, dtype='int64')
    for i in tqdm.tqdm(range(len(dataset))):
        rgb, labels = dataset.__getitem__(i, True)
        hist = np.histogram(
            labels.flatten(), bins=num_classes, range=(0, num_classes - 1))[0]
        hist_all += hist

    hist_all = hist_all.astype('float64')
    hist_all = hist_all / hist_all.sum()

    print(hist_all)
