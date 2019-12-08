import torch


def visualize_rgb(rgb, stats):
    # Scale back to [0,255]
    mean, std = stats
    device = rgb.get_device()
    mean = torch.tensor(mean).view(-1, 1, 1).to(device)
    std = torch.tensor(std).view(-1, 1, 1).to(device)

    rgb *= std[:3]
    rgb += mean[:3]
    return (255 * rgb).byte()


def iou_score(pred_cls, true_cls, nclass=14, drop=(0,)):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i).byte() +
                         (true_cls == i).byte()).eq(2).sum().item()
            union = ((pred_cls == i).byte() +
                     (true_cls == i).byte()).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return torch.tensor(intersect_), torch.tensor(union_)


def accuracy(pred_cls, true_cls, nclass=14, drop=(0,)):

    positive = torch.histc(true_cls.cpu().float(),
                           bins=nclass,
                           min=0,
                           max=nclass,
                           out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i).byte() +
                             (true_cls == i).byte()).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append(positive[i])

    return torch.tensor(tpos), torch.tensor(per_cls_counts)


def patches_to_pano_data(bilinear_resample_from_uv_layer,
                         nearest_resample_from_uv_layer,
                         rgb=None,
                         gt_labels=None,
                         pred_labels=None):
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
        rgb = torch.flip(rgb, (-2, )).contiguous()
        rgb = rgb.permute(1, 2, 0, 3, 4).contiguous()
        rgb = bilinear_resample_from_uv_layer(rgb)

    # Process GT
    if gt_labels is not None:
        gt_labels = torch.flip(gt_labels, (-2, )).contiguous()
        gt_labels = gt_labels.permute(1, 2, 0, 3, 4).contiguous()
        gt_labels = nearest_resample_from_uv_layer(
            gt_labels.float()).long()

    # Process outputs
    if pred_labels is not None:
        pred_labels = torch.flip(pred_labels, (-2, )).contiguous()
        pred_labels = pred_labels.permute(1, 2, 0, 3, 4).contiguous()
        pred_labels = nearest_resample_from_uv_layer(
            pred_labels.float()).long()

    return rgb, gt_labels, pred_labels


def pano_data_to_patches(bispherical_resample_to_texture,
                         nearest_resample_to_texture,
                         rgb=None,
                         gt=None):
    """
    rgb : B x 3 x H x W
    gt : B x 1 x H x W

    returns
        RGB: N x B x 3 x H x W
        gt: N x B x 1 x H x W
    """
    if rgb is not None:
        rgb = bispherical_resample_to_texture(rgb)
        rgb = rgb.permute(2, 0, 1, 3, 4).contiguous()
        rgb = torch.flip(rgb, (-2, )).contiguous()

    if gt is not None:
        gt = nearest_resample_to_texture(gt)
        gt = gt.permute(2, 0, 1, 3, 4).contiguous()
        gt = torch.flip(gt, (-2, )).contiguous()
    return rgb, gt
