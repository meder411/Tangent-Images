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

from mapped_convolution.util import IntrinsicsModifier


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
                 mean=None,
                 std=None):
        """Initialization"""
        self.data_path = omni_dp
        self.is_train = is_train
        self.data_format = data_format

        self.num_classes = 14
        self.ignore_label = 0

        self.scale_factor = scale_factor

        if self.is_train:
            extensions = ['SYNTHIA-SEQS-01-SUMMER',
                          'SYNTHIA-SEQS-02-SUMMER',
                          'SYNTHIA-SEQS-05-SUMMER',
                          'SYNTHIA-SEQS-06-SUMMER']
        else:
            extensions = ['SYNTHIA-SEQS-04-SUMMER']

        self.data_points = []

        if data_format == 'data':
            labels_folder = 'GT/LABELS'
            subst_index = -4
            template = os.path.join('RGB', 'Stereo_*', 'Omni_*', '*.png')
        else:
            labels_folder = 'GT/LABELS_Correct'
            subst_index = -2
            template = os.path.join('pano', 'RGB', '*.png')

        for ext in extensions:
            seq_template = os.path.join(self.data_path, ext, template)

            for full_im_path in glob.glob(seq_template):
                parts = full_im_path.split('/')
                parts[subst_index] = labels_folder
                labels_path = os.path.join(*parts)

                self.data_points.append((full_im_path, labels_path))

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])

        fx = fy = 532.7403520000000
        cx = 640
        cy = 380
        self.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.shift_x = 40
        self.shift_y = 20

        if data_format == 'data':
            self.bilinear_intrinsics_modifier = IntrinsicsModifier(
                *fov, *dim, 'bilinear')
            self.nearest_intrinsics_modifier = IntrinsicsModifier(
                *fov, *dim, 'nearest')

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index, to_cache=False):
        rgb_img_fp, labels_color_fp = self.data_points[index]
        basename = osp.splitext(osp.basename(rgb_img_fp))[0]
        if self.data_format == 'data':
            # uint16 image, labels are stored in the last channel
            labels = cv2.imread(labels_color_fp, cv2.IMREAD_UNCHANGED)[:, :, 2]
        else:
            # uint8 image, value=255 introduced for invalid region
            labels = cv2.imread(labels_color_fp, cv2.IMREAD_UNCHANGED)
            labels[labels == 255] = 0
        labels = labels.astype('float32')

        unq = np.unique(labels.tolist())
        assert not 13 in unq and not 14 in unq
        labels[labels == 15] = 13

        rgb = Image.open(rgb_img_fp)
        if to_cache:
            return rgb, labels
        rgb = self.preprocess(rgb)
        labels = torch.from_numpy(labels).unsqueeze(0)

        if self.data_format == 'data':
            K = torch.tensor(self.K).view(3, 3)
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
        hist = np.histogram(labels.flatten(), bins=num_classes,
                            range=(0, num_classes-1))[0]
        hist_all += hist

    hist_all = hist_all.astype('float64')
    hist_all = hist_all / hist_all.sum()

    print(hist_all)
