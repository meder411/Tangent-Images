import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images
from spherical_distortion.util import *

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import math
import os

# ------------
# Parameters
# ------------
sample_order = 10  # Determines sample resolution (10 = 2048 x 4096)
scale_factor = 1.0  # How much to scale input equirectangular image by


# --------------
# Draw function
# --------------
def draw_keypoints(fname, img, keypoints, color='r', linewidth=1):
    """Draws keypoints with scale and orientation (like OpenCV)"""

    # Radius
    r = kp[:, 2] / 2

    # Create circle objects
    circ = [
        patches.Circle(
            (kp[i, 0], kp[i, 1]),
            r[i],
            color=color,
            fill=False,
            linewidth=linewidth) for i in range(kp.shape[0])
    ]

    # Draw lines to show orientation (OpenCV uses clockwise rotation)
    cos_theta = torch.cos(-kp[:, 3])  # Cosine of angle
    sin_theta = torch.sin(-kp[:, 3])  # Sine of angle
    line_vec = torch.zeros(kp.shape[0], 2)
    line_vec[:, 0] = kp[:, 0] * cos_theta - kp[:, 1] * sin_theta
    line_vec[:, 1] = kp[:, 0] * sin_theta + kp[:, 1] * cos_theta
    line_vec = F.normalize(line_vec, dim=-1)

    # Create the lines for orientation
    lines = [
        mlines.Line2D(
            [kp[i, 0], kp[i, 0] + r[i] * line_vec[i, 0]],
            [kp[i, 1], kp[i, 1] + r[i] * line_vec[i, 1]],
            color=color,
            linewidth=linewidth) for i in range(kp.shape[0])
    ]

    # Set up the plot
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1, adjustable='box')
    ax.imshow(torch2numpy(img.byte()))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Plot the cicles and lines
    for (e, l) in zip(circ, lines):
        ax.add_patch(e)
        ax.add_line(l)

    fig.set_size_inches(12, 6)
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=500)


# ----------------
# Generate images
# ----------------
os.makedirs('outputs', exist_ok=True)

# Detect SIFT keypoints on the equirectangular image (-1) and base levels [0,2]
# Draw the result and save it to a PDF
for base_order in range(-1, 3):
    print('Base order:', base_order)

    img = load_torch_img('inputs/stanford-example.png', True)[..., :3]
    img = F.interpolate(
        img.float(),
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=False,
        recompute_scale_factor=True).squeeze(0)

    # If this is a tangent image run
    if base_order >= 0:

        # Corners of tangent planes in 3D coordinates
        corners = tangent_image_corners(base_order, sample_order)

        tan_img = create_tangent_images(img, base_order, sample_order).byte()
        kp, _ = sift_tangent_images(
            tan_img, corners, image_shape=img.shape[-2:], crop_degree=0)

    # If this is a equirectangular run
    else:
        kp, _ = sift_equirectangular(img, crop_degree=0)

    # Draw the keypoints and save it to a PDF file
    draw_keypoints('outputs/sift-detections_{}.pdf'.format(base_order), img,
                   kp, 'darkred', 0.5)