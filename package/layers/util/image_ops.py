import torch
import cv2
import math

from .conversions import *
import _tangent_images_ext._mesh as mesh


def compute_sift_keypoints(img,
                           nfeatures=0,
                           nOctaveLayers=3,
                           contrastThreshold=0.04,
                           edgeThreshold=10,
                           sigma=1.6):
    """
    Expects 3 x H x W torch tensor

    Returns [M x 4 (x, y, s, o), M x 128]
    """
    # Convert to numpy and ensure it's a uint8
    img = img.permute(1, 2, 0).byte().numpy()

    # Initialize OpenCV SIFT detector
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
    Visualize keypoints
    img: 3 x H x W
    keypoints: N x 4
    """
    kp = [cv2.KeyPoint(k[0], k[1], k[2], math.degrees(k[3])) for k in keypoints]
    out_img = cv2.drawKeypoints(
        img.permute(1, 2, 0).numpy(),
        kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return torch.from_numpy(out_img).permute(2, 0, 1)


def compute_crop(image_shape, crop_degree=0):
    """Compute padding space in an equirectangular images"""
    crop_h = 0
    if crop_degree > 0:
        crop_h = image_shape[0] // (180 / crop_degree)

    return crop_h


def sift_tangent_images(tex_image, corners, image_shape, crop_degree=0):
    """
    Extracts only the visible SIFT features from a collection tangent image. That is, only returns the keypoints visible to a spherical camera at the center of the icosahedron.

    tex_image: 3 x N x H x W
    corners: N x 4 x 3 coordinates of tangent image corners in 3D
    image_shape: (H, W) of equirectangular image that we render back to
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [visible_kp, visible_desc] (M x 4, M x 128)
    """

    # ----------------------------------------------
    # Compute SIFT descriptors for each patch
    # ----------------------------------------------
    kp_list = []  # Stores keypoint coords
    desc_list = []  # Stores keypoint descriptors
    quad_idx_list = []  # Stores quad index for each keypoint
    for i in range(tex_image.shape[1]):
        kp_details = compute_sift_keypoints(tex_image[:, i, ...])

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
    visible_kp = visible_kp[mask]  # M x 4
    visible_desc = visible_desc[mask]  # M x 128
    return visible_kp, visible_desc


def sift_equirectangular(img, crop_degree=0):
    """
    img: torch style (C x H x W) torch tensor
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [erp_kp, erp_desc] (M x 4, M x 128)
    """

    # ----------------------------------------------
    # Compute SIFT descriptors on equirect image
    # ----------------------------------------------
    erp_kp_details = compute_sift_keypoints(img)
    erp_kp = erp_kp_details[0]
    erp_desc = erp_kp_details[1]

    # If top top and bottom of image is padding
    crop_h = compute_crop(img.shape[-2:], crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (erp_kp[:, 1] > crop_h) & (erp_kp[:, 1] < img.shape[1] - crop_h)
    erp_kp = erp_kp[mask]
    erp_desc = erp_desc[mask]

    return erp_kp, erp_desc