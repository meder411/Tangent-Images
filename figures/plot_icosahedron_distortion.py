import torch
from tangent_images.util import *
import matplotlib.pyplot as plt
import numpy as np

# Max level
max_level = 10

# Lists for the data
level = []
ratio = []

# Generate data
for i in range(max_level + 1):
    m = generate_icosphere(i)
    m.normalize_points()
    level.append(i)
    ratio.append(m.surface_area() / (4 * np.pi * (m.radius()**2)))

# Plot it
fig, ax = plt.subplots(1, 1)
ax.grid(False)
ax.set_xlabel('Subdivision Level', fontsize=28)
ax.set_ylabel('Surface Area Ratio', fontsize=28)
ax.plot(np.array(level), np.array(ratio), '-bo', linewidth=4, markersize=10)
plt.xticks(np.arange(0, 11, step=1))
plt.axvline(x=3, color='lightcoral', linewidth=4)  # Verticsl line

# Save figure to high quality PDF
fig.set_size_inches(12, 6)
fig.savefig(
    'icosahedron_distortion.pdf', bbox_inches='tight', pad_inches=0, dpi=500)

# Also show the figure
plt.show()