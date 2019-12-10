import torch.nn.functional as F
from tangent_images.nn import Resample
from .grids import *  # Comment when running as a script
# from grids import *  # Uncomment when running as a script


class BrownDistortion(object):

    def __init__(self, k1, k2, k3, t1, t2, crop=True):

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.t1 = t1
        self.t2 = t2
        self.crop = crop
        self.resample = Resample('bilinear')

    def __call__(self, img):
        """
        img: N x H x W tensor
        """
        # Check if batch dim is present
        no_batch_dim = False
        if img.dim() == 3:
            no_batch_dim = True

        # Compute the distortion grid
        H, W = img.shape[-2:]
        dgrid_y, dgrid_x = self._compute_distortion_grid((H, W))

        # Compute the unnormalized distortion grid (i.e the sample map)
        un_dgrid_y, un_dgrid_x = self._unnormalize_distortion_grid(
            (H, W), dgrid_y, dgrid_x)

        # Get dims of output tensor
        outW = un_dgrid_x.max().ceil().long()
        outH = un_dgrid_y.max().ceil().long()

        # Create sample map
        sample_map = torch.stack((un_dgrid_x, un_dgrid_y), -1)
        if torch.cuda.is_available() and img.get_device() >= 0:
            sample_map = sample_map.to(img.get_device())

        # Add the batch dimension for resampling
        if no_batch_dim:
            img = img.unsqueeze(0)

        # Resample to output image tensor
        output_img = self.resample(img, sample_map, (outH, outW))

        # Normalization hack
        norm_img = self.resample(torch.ones(1, 1, H, W), sample_map,
                                 (outH, outW))

        # Normalize
        output_img /= norm_img

        if no_batch_dim:
            output_img.squeeze_(0)

        # Center crop if desired
        if self.crop and (outH > H) and (outW > W):
            diffH = outH - H
            diffW = outW - W
            return output_img[..., diffH // 2:diffH // 2 + H, diffW //
                              2:diffW // 2 + W]

        if (outH < H) or (outW < W):
            diffH = H - outH
            diffW = W - outW
            halfH = diffH // 2
            halfW = diffW // 2
            output_img = F.pad(
                output_img, (halfH, H - outH - halfH, halfW, W - outW - halfW))

        return output_img

    def _unnormalize_distortion_grid(self, shape, dgrid_y, dgrid_x):
        """
        Un-normalized the distortion grid using the original dimensions. Note that this can result in an image larger than the original.
        """

        # Precompute some useful values
        H, W = shape
        half_width = ((W - 1) / 2)
        half_height = ((H - 1) / 2)

        # Unnormalize the grid
        dgrid_x *= half_width
        dgrid_y *= half_height

        # Uncenter the grid s.t top-left pixel is (0,0)
        dgrid_x -= dgrid_x.min()
        dgrid_y -= dgrid_y.min()

        # Return the unnormalized distortion grid
        return dgrid_y, dgrid_x

    def _compute_distortion_grid(self, shape):
        """
        shape: (H, W)
        Computes and returns the distortion grid
        """
        grid_y, grid_x = normalized_meshgrid(shape)

        # Radial terms
        r2 = (grid_x**2) + (grid_y**2)
        r4 = r2 * r2
        r6 = r2 * r4

        # Compute terms of distortion function
        k_diff = self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        t_x = self.t2 * (r2 + 2 * (grid_x**2)) + 2 * self.t1 * grid_x * grid_y
        t_y = self.t1 * (r2 + 2 * (grid_y**2)) + 2 * self.t2 * grid_x * grid_y

        # Compute the distortion grids
        dist_grid_x = grid_x * k_diff + t_x
        dist_grid_y = grid_y * k_diff + t_y

        # Add the distortion grids to the original coordinates
        grid_x += dist_grid_x
        grid_y += dist_grid_y

        # Return the distortion grid
        return grid_y, grid_x


if __name__ == '__main__':

    # from mapped_convolution.util import *
    from skimage import io
    import torch
    import torch.nn.functional as F

    k1 = -0.4
    k2 = 0.0
    k3 = 0.0
    t1 = 0.0
    t2 = 0.0
    img = io.imread(
        # '/home/meder/Research/mapped_convolutions/package/layers/util/headshot.jpg'
        # '/home/meder/Research/mapped_convolutions/package/mnist.jpg')
        '/home/meder/Research/mapped_convolutions/package/mnist.png')
    img = torch.from_numpy(img)[..., [0]].permute(2, 0, 1).float()
    img = F.interpolate(
        img.unsqueeze(0),
        size=(28, 28),
        # scale_factor=0.1,
        mode='bilinear',
        align_corners=False).squeeze(0)

    # # Temporary
    # # Pad the image according to inverse radial distortion
    # H, W = img.shape[-2:]
    # print(img.shape[-2:])
    # img = F.pad(
    #     img,
    #     ((2500 - H) // 2, (2500 - H) // 2, (1500 - W) // 2, (1500 - W) // 2),
    # )
    # print(img.shape[-2:])

    bdist = BrownDistortion(k1, k2, k3, t1, t2)
    # ipdb.set_trace()
    print(img.shape)
    out = bdist(img)
    print(out.shape)
    io.imsave('test_dist{}.png'.format(k1), out.byte().squeeze().numpy())
    # io.imsave('test_dist{}.png'.format(k1), out.byte().permute(1, 2, 0).numpy())