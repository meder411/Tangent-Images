from spherical_distortion.util import *

sample_order = 9 # Input resolution to examine

def ang_fov(s):
    print('Spherical Resolution:', s)
    for b in range(s):
        dim = tangent_image_dim(b, s) # Pixel dimension of tangent image
        corners = tangent_image_corners(b, s) # Corners of each tangent image
        fov_x, fov_y = compute_tangent_image_angular_resolution(corners)
        print('  At base level', b)
        print('    FOV (x) =', fov_x)
        print('    FOV (y) =', fov_y)
        print('    deg/pix (x) =', fov_x/dim)
        print('    deg/pix (y) =', fov_y/dim)

ang_fov(sample_order)