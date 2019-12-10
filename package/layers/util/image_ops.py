import torch
import cv2

from .conversions import *
import _tangent_images_ext._mesh as mesh
import math


def rgb_to_gray(images):
    """
    images: * x ... x * x C x H x W tensor containing RGB images

    returns * x ... x * x H x W grayscale images
    """
    return 0.299 * images[..., 0, :, :] + 0.587 * images[
        ..., 1, :, :] + 0.114 * images[..., 2, :, :]


def compute_sift_keypoints(img,
                           nfeatures=0,
                           nOctaveLayers=3,
                           contrastThreshold=0.04,
                           edgeThreshold=10,
                           sigma=1.6):
    """
    Expects H x W x C RGB image

    Returns [M x 4 (x, y, s, o), M x 128]
    """

    sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers,
                                       contrastThreshold, edgeThreshold, sigma)

    # Keypoints is a list of lenght N, desc is N x 128
    keypoints, desc = sift.detectAndCompute(img, None)
    if len(keypoints) > 0:
        coords = torch.tensor([kp.pt for kp in keypoints])
        orientation = torch.tensor(
            [kp.angle * math.pi / 180 for kp in keypoints])
        scale = torch.tensor([kp.size for kp in keypoints])
        desc = torch.from_numpy(desc)
        return torch.cat((coords, scale.unsqueeze(1), orientation.unsqueeze(1)),
                         -1), desc
    else:
        return None


def compute_visible_keypoints(kp_quad_idx, kp_details, kp_desc, quad_corners,
                              quad_shape):
    """
    kp_quad_idx: M
    kp_details: M x 4
    kp_desc: M x 128
    quad_corners: N x 4 x 3
    quad_shape: (H, W)

    returns [K x 3, K x 128, K, K]
    """

    # Convert the quad coordinates to normalized UV coords
    kp_uv = convert_quad_coord_to_uv(quad_shape, kp_details[:, :2].contiguous())

    # Convert the normalized quad UV data to 3D points
    kp_3d = convert_quad_uv_to_3d(kp_quad_idx, kp_uv, quad_corners)

    # Find the points visible to a spherical camera
    valid_kp_3d, valid_desc, valid_kp_scale, valid_kp_orient = mesh.find_visible_keypoints(
        kp_3d.float(), kp_quad_idx, kp_desc.float(),
        kp_details[:, 2].contiguous().float(),
        kp_details[:, 3].contiguous().float(), quad_corners.float())

    return valid_kp_3d, valid_desc, valid_kp_scale, valid_kp_orient


def render_keypoints(image_shape, kp_quad_idx, kp_details, kp_desc,
                     quad_corners, quad_shape):
    """
    image_shape: (H, W)
    kp_quad_idx: M
    kp_details: M x 4
    kp_desc: M x 128
    quad_corners: N x 4 x 3
    quad_shape: (H, W)

    return: [K x 2, K x 128]
    """
    # Compute the visible keypoints
    valid_kp_3d, valid_desc, valid_kp_scale, valid_kp_orient = compute_visible_keypoints(
        kp_quad_idx, kp_details, kp_desc, quad_corners, quad_shape)

    # Convert to equirectangular image coordinates
    valid_kp_img = convert_spherical_to_image(
        convert_3d_to_spherical(valid_kp_3d), image_shape)

    return torch.cat((valid_kp_img, valid_kp_scale.unsqueeze(1),
                      valid_kp_orient.unsqueeze(1)), -1), valid_desc


def draw_keypoints(img, keypoints):
    """
    VISUALIZE KEYPOINTS
    img: H x W x 3
    keypoints: N x 4
    """
    kp = [cv2.KeyPoint(k[0], k[1], k[2], math.degrees(k[3])) for k in keypoints]
    out_img = cv2.drawKeypoints(
        img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return out_img


def compute_crop(image_shape, crop_degree=0):
    """Compute padding space"""
    if crop_degree > 0:
        crop_h = image_shape[0] // (180 / crop_degree)
    else:
        crop_h = 0

    return crop_h


def extract_sift_feats_patch(tex_image, corners, image_shape, crop_degree=0):

    # ----------------------------------------------
    # Compute SIFT descriptors for each patch
    # ----------------------------------------------
    kp_list = []  # Stores keypoint coords
    desc_list = []  # Stores keypoint descriptors
    quad_idx_list = []  # Stores quad index for each keypoint
    for i in range(tex_image.shape[1]):
        # ipdb.set_trace()
        kp_details = compute_sift_keypoints(tex_image[:, i, ...].permute(
            1, 2, 0).numpy())

        if kp_details is not None:
            kp = kp_details[0]
            desc = kp_details[1]
            kp_list.append(kp)
            desc_list.append(desc)
            quad_idx_list.append(i * torch.ones(kp.shape[0]))

    # Assemble keypoint data
    patch_kp = torch.cat(kp_list, 0).float()  # M x 4 (x, y, s, o)
    patch_desc = torch.cat(desc_list, 0).float()  # M x 128
    patch_quad_idx = torch.cat(quad_idx_list, 0).long()  # M

    # ----------------------------------------------
    # Compute only visible keypoints
    # ----------------------------------------------
    visible_kp, visible_desc = render_keypoints(image_shape, patch_quad_idx,
                                                patch_kp, patch_desc, corners,
                                                tex_image.shape[-2:])

    # If top top and bottom of image is padding
    crop_h = compute_crop(image_shape, crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (visible_kp[:, 1] > crop_h) & (visible_kp[:, 1] <
                                          image_shape[0] - crop_h)
    visible_kp = visible_kp[mask]
    visible_desc = visible_desc[mask]
    return visible_kp


def extract_sift_feats_erp(img, crop_degree=0):
    """
    img: torch style (C x H x W) torch tensor
    """

    # ----------------------------------------------
    # Compute SIFT descriptors on equirect image
    # ----------------------------------------------
    erp_kp_details = compute_sift_keypoints(img.permute(1, 2, 0).numpy())
    erp_kp = erp_kp_details[0]
    erp_desc = erp_kp_details[1]

    # If top top and bottom of image is padding
    crop_h = compute_crop(img.shape[-2:], crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (erp_kp[:, 1] > crop_h) & (erp_kp[:, 1] < img.shape[1] - crop_h)
    erp_kp = erp_kp[mask]
    erp_desc = erp_desc[mask]

    return erp_kp