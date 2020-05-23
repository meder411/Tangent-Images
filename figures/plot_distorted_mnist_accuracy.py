import numpy as np
import matplotlib.pyplot as plt

# Results file
baseline_results_file = '../experiments/distorted_mnist/baseline-accuracy.txt'
aug_results_file = '../experiments/distorted_mnist/aug-accuracy.txt'


# Load results
def load_results(results_file):
    results = np.loadtxt(results_file)
    radial_distortions = results[:, 0]
    accuracies = results[:, 1]
    return radial_distortions, accuracies


radial_distortions, base_acc = load_results(baseline_results_file)
_, aug_acc = load_results(aug_results_file)

# Plot the results
fig, ax = plt.subplots(1, 1)
ax.grid(False)
ax.set_xlabel('K1', fontsize=24)
ax.set_ylabel('Accuracy', fontsize=24)
ax.set_xticks(radial_distortions[0::10] / 100)
ax.tick_params(axis='both', which='major', labelsize=16)

ax.plot(radial_distortions / 100, base_acc, '-b', linewidth=4, markersize=10)
# ax.plot(radial_distortions / 100, aug_acc, '-r', linewidth=4, markersize=10)
# ax.legend(['Without Augmentation', 'With Augmentation'], fontsize=16)

fig.set_size_inches(12, 6)
fig.savefig(
    'distorted_mnist_results.pdf', bbox_inches='tight', pad_inches=0, dpi=500)
plt.show()