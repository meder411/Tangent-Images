import numpy as np
import matplotlib.pyplot as plt

# Results file
results_file = '../experiments/distorted_mnist/accuracy.txt'

# Load results
results = np.loadtxt(results_file)
radial_distortions = results[:, 0]
accuracies = results[:, 1]

# Fit a line to the results
z = np.polyfit(radial_distortions, accuracies, 3)
p = np.poly1d(z)

# Plot the results
fig, ax = plt.subplots(1, 1)
ax.grid(False)
ax.set_xlabel('K1', fontsize=28)
ax.set_ylabel('Accuracy', fontsize=28)
ax.set_xticks(radial_distortions[0::10])
ax.plot(
    radial_distortions, p(radial_distortions), '-b', linewidth=4, markersize=10)

fig.set_size_inches(12, 7)
fig.savefig(
    'distorted_mnist_results.pdf', bbox_inches='tight', pad_inches=0, dpi=500)
plt.show()