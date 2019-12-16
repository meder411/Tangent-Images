import torch
import torch.nn as nn

import math

from tangent_images.nn import Unresample


class IntrinsicsModifier(nn.Module):
    """Resamples an image to a different intrinsics matrix"""

    def __init__(self,
                 out_fov_x,
                 out_fov_y,
                 out_dim_x,
                 out_dim_y,
                 interpolation='bilinear'):
        """
        out_fov_x: degrees
        out_fov_y: degrees
        out_dim_x: pixels
        out_dim_y: pixels
        """

        super(IntrinsicsModifier, self).__init__()

        # Store output dimensions
        self.out_dim_x = out_dim_x
        self.out_dim_y = out_dim_y
        self.out_fov_x = out_fov_x  # degrees
        self.out_fov_y = out_fov_y  # degrees

        # Create the output grid
        y, x = torch.meshgrid(torch.arange(out_dim_y), torch.arange(out_dim_x))
        self.grid = torch.stack(
            (x.float(), y.float(), torch.ones(out_dim_y, out_dim_x)), -1)

        # Compute the new camera matrix
        self.fx = out_dim_x / (2 * math.tan(math.radians(out_fov_x) / 2))
        self.fy = out_dim_y / (2 * math.tan(math.radians(out_fov_y) / 2))
        self.cx = out_dim_x / 2
        self.cy = out_dim_y / 2
        self.new_Kinv_T = torch.tensor(
            [[1 / self.fx, 0, 0], [0, 1 / self.fy, 0],
             [-self.cx / self.fx, -self.cy / self.fy, 1]])

        self.unresample = Unresample(interpolation)

    def invert_K(self, fx, fy, cx, cy):
        # Inverts the passed in camera matrix
        return torch.tensor([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy],
                             [0, 0, 1]])

    def make_sample_grid(self, old_K):
        # Returns a dim x dim x 2 sample grid

        return (self.grid.view(-1, 3) @ self.new_Kinv_T @ old_K.T).view(
            self.out_dim_x, self.out_dim_y, 3)[..., :2].contiguous()

    def forward(self, img, K, out_shift_x=0, out_shift_y=0):
        """
        img: B x C x H x W image
        K: 3 x 3 intrinsics matrix
        out_shift_x: scalar shift of the camera center of the passed in intrinsics. + shifts right, - shifts left. In terms of pixels of the passed in image.
        out_shift_y: scalar shift of the camera center of the passed in intrinsics. + shifts down, - shifts up. In terms of pixels of the passed in image.
        """

        # Shift the camera
        K[0, -1] += out_shift_x
        K[1, -1] += out_shift_y

        # Create a sample map to unresample to
        sample_grid = self.make_sample_grid(K)

        # Unresample the image
        return self.unresample(img, sample_grid)


if __name__ == '__main__':

    from skimage import io
    import torch
    import torch.nn.functional as F

    # Note: it's a square image
    img = io.imread(
        # '/home/meder/Research/mapped_convolutions/package/000000.png')
        '/home/meder/Research/mapped_convolutions/package/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_0_domain_rgb.png'
    )
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    K = torch.tensor([[759.6317749023438, 0.0, 540.0],
                      [0.0, 759.6317749023438, 540.0], [0.0, 0.0, 1.0]])
    # K = torch.tensor([[532.740352, 0.0, 640.0], [0.0, 532.740352, 380.0],
    # [0.0, 0.0, 1.0]])
    out_fov_x = 45
    out_fov_y = 45
    out_dim_x = 128
    out_dim_y = 128
    out_shift_x = 0
    out_shift_y = 100
    modifier = IntrinsicsModifier(out_fov_x, out_fov_y, out_dim_x, out_dim_y)

    out = modifier(img, K, out_shift_x, out_shift_y)
    io.imsave('test_cam_norm.png',
              out.squeeze().byte().permute(1, 2, 0).numpy())