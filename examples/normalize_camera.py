import torch
import time
from spherical_distortion.util import load_torch_img, torch2numpy
from spherical_distortion.transforms import CameraNormalization
import numpy as np
from skimage import io
import os

# Load the image and it's associated K matrix
os.makedirs('outputs', exist_ok=True)
img = load_torch_img('inputs/synthia-car.png').float()
K = torch.from_numpy(np.loadtxt('inputs/synthia-car-intrinsics.txt')).float()

# Set the output parameters
fov_out = (45, 45)
shape_out = (128, 128)
random_crop = True

# Initialize the camera normalization transform
transform = CameraNormalization(fov_out, shape_out, random_crop)

# Time the operation and print some feedback
print('Input Shape:', img.shape)
s = time.time()
out = transform(img, K)
print('Time:', time.time() - s)
print('New K:', transform.get_K())
print('Output Shape:', out.shape)

# Save the result
os.makedirs('outputs', exist_ok=True)
io.imsave('outputs/normalized-synthia-car.png', torch2numpy(out.byte()))