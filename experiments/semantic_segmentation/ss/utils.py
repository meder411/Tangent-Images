import torch
from spherical_distortion.util import InterpolationType
from spherical_distortion.functional import uv_resample, unresample


def visualize_rgb(rgb, stats):
    # Scale back to [0,255]
    mean, std = stats
    device = rgb.get_device()
    mean = torch.tensor(mean).view(-1, 1, 1).to(device)
    std = torch.tensor(std).view(-1, 1, 1).to(device)

    rgb *= std[:3]
    rgb += mean[:3]
    return (255 * rgb).byte()


def tan2rect(quad_map, uv_map, rgb=None, gt_labels=None, pred_labels=None):
    """
    RGB: N x B x 3 x H x W
    gt_labels: N x B x 1 x H x W
    pred_labels: N x B x 1 x H x W

    returns
        rgb : B x 3 x H x W
        gt_labels : B x 1 x H x W
        pred_labels : B x 1 x H x W
    """

    # Process RGB
    if rgb is not None:
        rgb = rgb.permute(1, 2, 0, 3, 4).contiguous()
        rgb = uv_resample(rgb, quad_map, uv_map, InterpolationType.BILINEAR)

    # Process GT
    if gt_labels is not None:
        gt_labels = gt_labels.permute(1, 2, 0, 3, 4).contiguous()
        gt_labels = uv_resample(gt_labels.float(), quad_map, uv_map,
                                InterpolationType.NEAREST).long()

    # Process outputs
    if pred_labels is not None:
        pred_labels = pred_labels.permute(1, 2, 0, 3, 4).contiguous()
        pred_labels = uv_resample(pred_labels.float(), quad_map, uv_map,
                                  InterpolationType.NEAREST).long()

    return rgb, gt_labels, pred_labels


def rect2tan(sample_map, tan_dim, rgb=None, gt=None):
    """
    rgb : B x 3 x H x W
    gt : B x 1 x H x W

    returns
        RGB: N x B x 3 x H x W
        gt: N x B x 1 x H x W
    """
    if rgb is not None:
        rgb = unresample(rgb, sample_map, InterpolationType.BISPHERICAL)
        rgb = rgb.view(*rgb.shape[:-1], tan_dim, tan_dim)
        rgb = rgb.permute(2, 0, 1, 3, 4).contiguous()

    if gt is not None:
        gt = unresample(gt, sample_map, InterpolationType.NEAREST)
        gt = gt.view(*gt.shape[:-1], tan_dim, tan_dim)
        gt = gt.permute(2, 0, 1, 3, 4).contiguous()
    return rgb, gt
