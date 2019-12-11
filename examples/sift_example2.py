import torch
import torch.nn.functional as F
from mapped_convolution.util import *
from mapped_convolution.nn import *

from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import math


def draw_keypoints(fname, img, keypoints, color='r', linewidth=1):

    # Radius
    r = kp[:, 2] / 2

    # Create circle objects
    circ = [
        patches.Circle(
            (kp[i, 0], kp[i, 1]),
            r[i],
            color=color,
            # alpha=0.5,
            fill=False,
            linewidth=linewidth) for i in range(kp.shape[0])
    ]

    # Draw lines to show angle (OpenCV uses clockwise rotation)
    cos_theta = torch.cos(-kp[:, 3])  # Cosine of angle
    sin_theta = torch.sin(-kp[:, 3])  # Sine of angle
    line_vec = torch.zeros(kp.shape[0], 2)
    line_vec[:, 0] = kp[:, 0] * cos_theta - kp[:, 1] * sin_theta
    line_vec[:, 1] = kp[:, 0] * sin_theta + kp[:, 1] * cos_theta
    line_vec = F.normalize(line_vec, dim=-1)

    lines = [
        mlines.Line2D([kp[i, 0], kp[i, 0] + r[i] * line_vec[i, 0]],
                      [kp[i, 1], kp[i, 1] + r[i] * line_vec[i, 1]],
                      color=color,
                      linewidth=linewidth) for i in range(kp.shape[0])
    ]

    # Set up the plot
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1, adjustable='box')
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Plot the cicles and lines
    for (e, l) in zip(circ, lines):
        ax.add_patch(e)
        ax.add_line(l)

    fig.set_size_inches(12, 6)
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=500)


scale = 1.0
sample_level = 10

for sift_type in range(-1, 3):
    print('Sift type:', sift_type)
    img = io.imread('right-img.png')[..., :3]
    if scale < 1.0:
        img = (255 * transform.rescale(img.astype(np.float32) / 255,
                                       scale=scale,
                                       order=1,
                                       anti_aliasing=True,
                                       mode='constant',
                                       multichannel=True)).astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1)

    # If this is a patch run
    if sift_type >= 0:
        resample_to_uv_layer, corners = get_tangent_plane_info(
            sift_type, sample_level, img.shape[-2:])
        tex_image = resample_to_uv_layer(
            img.float().unsqueeze(0)).squeeze(0).byte()
        kp = extract_sift_feats_patch(tex_image,
                                      corners,
                                      image_shape=img.shape[-2:],
                                      crop_degree=30)
    else:
        kp = extract_sift_feats_erp(img, crop_degree=30)

    draw_keypoints('sift-detections_3_{}.pdf'.format(sift_type), img, kp,
                   'darkred', 0.5)