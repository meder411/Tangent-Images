# 3D Shape Classification on [ModelNet40](https://modelnet.cs.princeton.edu/)

## Results

Tangent images allow us to achieve near state-of-the-art performance (12/2019) without any specialized filters.

| Method | Filter | Acc. |
|--------|--------|------|
| Cohen *et al.* [9] | Spherical Correlation | 85.0%  |
| Esteves *et al.* [13] | Spectral Parameterization | 88.9%  |
| Jiang *et al.* [17] | MeshConv | 90.5%  |
| **Ours** | **2D Convolution** | **89.1%**  |


## Attribution
The code for this experiment is modified from [UGSCNN](https://github.com/maxjiang93/ugscnn/tree/master/experiments/exp2_modelnet40). Our modifications are primarily to the model itself and the dataloader, so that we can evaluate tangent images.


## Dependencies
The code requires the following dependencies that can be installed using conda and pip.

```
conda install -c conda-forge rtree shapely  
conda install -c conda-forge pyembree  
pip install "trimesh[easy]"  
```

**Note:** if you are getting errors installing `rtree` or `trimesh`, try first installing:

`sudo apt install libspatialindex-dev`

It seems to be a dependency for the above Python libraries.

## Usage
Train a model using the training script:

```
chmod +x train.sh
./train.sh
```

Test a trained model (e.g. the included `best.pkl`) with the test script:

```
chmod +x test.sh
./test.sh
```

Both scripts will automatically download the data files if they do not already exist.