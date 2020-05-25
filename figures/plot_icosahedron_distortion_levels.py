import torch
from spherical_distortion.util import *
import matplotlib.pyplot as plt
import numpy as np
import os

# Max subdivision level
max_order = 10

# Lists for the data
level = []
ratio = []

# Generate each icosahedron and compute the surface area ratio to the sphere
for i in range(max_order + 1):
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
plt.xticks(np.arange(0, max_order + 1, step=1))

# Plot the vertical line where distortion levels off
plt.axvline(x=3, color='lightcoral', linewidth=4)

# Save figure to high quality PDF
os.makedirs('outputs', exist_ok=True)
fig.set_size_inches(12, 6)
fig.savefig(
    'outputs/icosahedron-distortion.pdf',
    bbox_inches='tight',
    pad_inches=0,
    dpi=300)

# Also show the figure
plt.show()