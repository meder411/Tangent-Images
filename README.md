# Tangent-Images
![Tangent Images](images/figure_1.png)

## Dependencies

This repository is designed to be used with PyTorch. In its current form, it has been tested with PyTorch 1.3.1.

#### Python

* PyTorch 1.3.1
* numpy
* matplotlib (plotting examples)
* scikit-image (io)
* plyfile (for saving pointclouds)
* visdom (training visualizations)
* OpenCV 3.4.2 (note that this version is required for proper SIFT detection)

#### Other dependencies

* CGAL 4.11+ (the aptitude package version works)
* CUDA 10.1 (optional for PyTorch GPU support)


## Installation

#### Installing dependencies
To install the necessary Python dependencies, we recommend creating a Conda environment to install from the provided `tangent-images.yml` file. [Directions can be found here.](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

CGAL is a required dependency. You can [install the most recent version from source](https://doc.cgal.org/latest/Manual/general_intro.html) (header only is fine). If you use the aptitude package manager on Linux, you can alternatively install the library with the command:

```
sudo apt install libcgal-dev
```

#### Installing tangent-images

 1. Make sure you have activated your Conda environment or are otherwise using the desired Python environment
 2. Navigate to `<top-level>/package`
 3. `python setup.py install`

You should be able to test if the installation was successful by running the example scripts in [examples](./examples)

## Experiments

All experiments are included in the [experiments](./experiments) folder. Each experiment subdirectory has a README file explaining how to setup and run each experiments. Where relevant, we have included the pre-trained models corresponding to our published results.

## Attribution

If you find this repository useful for your own work, please make sure to cite our paper:

TBD.