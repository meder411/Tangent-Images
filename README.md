# Tangent Images for Mitigating Spherical Distortion
**Version: 1.0.0**

![Creating tangent images](./images/tangent-images.png)
<img src="./images/tangent-images.gif" width="250">

This repository contains the code corresponding to our CVPR 2020 paper: [Tangent Images for Mitigating Spherical Distortion](https://arxiv.org/abs/1912.09390). 

The linked paper is the arXiv version, which we have updated with additional experiments and which contains results that matches the code provided here.


## Dependencies

This repository is designed to be used with PyTorch. This code requires the installation of my Spherical Distortion Package, which is submoduled in this repository. [Installation instructions and/or a Dockerfile can be found here](https://github.com/meder411/spherical-package).

You should be able to test if the installation was successful by running the example scripts in [examples](./examples).


## Examples

In the [examples](./examples) directory, we have provided some basic examples to help you get started using tangent images. These examples include rendering to and from tangent images as well some visualizations of tangent images for SIFT keypoint and Canny edge detection.


## Experiments

All experiments are included in the [experiments](./experiments) folder. Each experiment subdirectory has a README file explaining how to setup and run each experiment. Where relevant, we have included the pre-trained models corresponding to our published results.


## Attribution

If you find this repository useful for your own work, please make sure to cite our paper:

```
@article{eder2019tangent,
    title={Tangent Images for Mitigating Spherical Distortion},
    author={Marc Eder and Mykhailo Shvets and John Lim and Jan-Michael Frahm},
    eprint={arXiv:1912.09390},
    year={2019}
}
```