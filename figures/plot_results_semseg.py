import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([5])
y0acc = np.array([55.9])
y0iou = np.array([39.4])
x1 = np.array([5])
y1acc = np.array([54.7])
y1iou = np.array([38.3])
x2 = np.array([5])
y2acc = np.array([58.6])
y2iou = np.array([43.3])
x3 = np.array([5, 7, 10])
y3acc = np.array([50.2, 54.9, 70.0])  # first 2 are TMP
y3iou = np.array([37.5, 41.8, 52.6])  # first 2 are TMP

fig, ax = plt.subplots(1, 1)
ax.grid(False)
ax.set_xlabel('Resolution Level', fontsize=28)
ax.set_ylabel('Metric', fontsize=28)
ax.plot(x0, y0acc, 'r+', x0, y0iou, 'b+', markersize=10)
ax.plot(x1, y1acc, 'r*', x1, y1iou, 'b*', markersize=10)
ax.plot(x2, y2acc, 'ro', x2, y2iou, 'bo', markersize=10)
ax.plot(x3, y3acc, '-r^', x3, y3iou, '-b^', linewidth=4, markersize=10)
ax.legend(
    [
        '[8] mAcc.', '[8] mIOU', '[16] mAcc.', '[16] mIOU', '[27] mAcc.',
        '[27] mIOU', 'Ours mAcc.', 'Ours mIOU'
    ],
    fontsize=14,
    loc=2)
plt.xticks(np.arange(3, 12, step=1))
plt.yticks(np.arange(25, 75, step=15))

# Save figure to high quality PDF
fig.set_size_inches(12, 7)
fig.savefig(
    'semantic_segmentation_results.pdf',
    bbox_inches='tight',
    pad_inches=0,
    dpi=500)

plt.show()