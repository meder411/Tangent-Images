import numpy as np
import matplotlib.pyplot as plt

# Average results of 3 folds using L8 angular normalized model
levels = np.array([5, 6, 7, 8, 9, 10])
iou = np.array([6.9, 16.7, 30.1, 35.6, 30.0, 18.8])
acc = np.array([23.9, 36.1, 47.1, 51.2, 45.9, 36.9])

# Plot settings
width = 0.45
fig, ax = plt.subplots(1, 1)
ax.grid(False)
ax.set_xlabel('Resolution Level', fontsize=28)
ax.set_ylabel('Metric', fontsize=28)
ax.set_xticks(levels + width / 2)
ax.set_xticklabels(levels)

# Bar plots side-by-side
pl_iou = ax.bar(levels, iou, width=width, color='cornflowerblue')
pl_acc = ax.bar(levels + width, acc, width=width, color='darksalmon')
pl_iou_l8 = ax.bar(
    levels[levels == 8], iou[levels == 8], width=width, color='navy')
pl_acc_l8 = ax.bar(
    levels[levels == 8] + width, acc[levels == 8], width=width, color='maroon')

# Annotate the value
# zip joins x and y coordinates in pairs
for l, a, i in zip(levels, acc, iou):

    acc_label = '{}'.format(a)
    iou_label = '{}'.format(i)

    # Add accuracy label above bars
    plt.annotate(
        acc_label, (l + width, a),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center')

    # Add IOU label above bars
    plt.annotate(
        iou_label, (l, i),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center')

ax.legend((pl_acc, pl_iou), ['mAcc.', 'mIOU'], fontsize=20, loc=2)

fig.set_size_inches(12, 10)
fig.savefig(
    'camera_normalization_plot.pdf', bbox_inches='tight', pad_inches=0, dpi=500)
plt.show()