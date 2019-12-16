## Examples

This directory contains scripts to generate tangent images and detect SIFT keypoints on them. These example scripts should help clarify how to use this library.

* **generate_tangent_images.py**: Turns an equirectangular images into tangent image patches
* **visualize_icosphere_sampling.py**: Visualizes tangent images alongside the icosahedron as well as the sampling points on the sphere
* **create_tangent_image_obj.py**: Writes an OBJ file of tangent images textured with the data from an equirectangular image
* **compare_sift_keypoints.py**: Detects SIFT keypoints on both an equirectangular image and tangent images and visualizes a comparison between them.
* **draw_sift_keypoints.py**: Detects SIFT keypoints on both an equirectangular image and tangent images with base levels [0,2], and draws the results with scale and orientation. Saves the outputs to PDF files.