import numpy as np
import matplotlib.pyplot as plt

# Results file
results_file = '../experiments/distorted_mnist/accuracy.txt'


# Load results
def load_results(results_file):
    results = np.loadtxt(results_file)
    radial_distortions = results[:, 0]
    accuracies = results[:, 1]
    return radial_distortions, accuracies


radial_distortions, acc = load_results(results_file)

# Plot the results
fig, ax = plt.subplots(1, 1)
ax.grid(False)
ax.set_xlabel('K1', fontsize=24)
ax.set_ylabel('Accuracy', fontsize=24)
ax.set_xticks(radial_distortions[0::10] / 100)
ax.tick_params(axis='both', which='major', labelsize=16)

ax.plot(radial_distortions / 100, acc, '-b', linewidth=4, markersize=10)

fig.set_size_inches(12, 6)
fig.savefig(
    'distorted_mnist_results.pdf', bbox_inches='tight', pad_inches=0, dpi=500)
plt.show()