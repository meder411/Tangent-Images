## 3D Shape Classification on ModelNet40

## Results

Tangent images allow us to achieve near state-of-the-art performance (12/2019) without any specialized filters.

| Method | Filter | Acc. |
|--------|--------|------|
| Cohen *et al.* [9] | Spherical Correlation | 85.0%  |
| Esteves *et al.* [13] | Spherical Correlation | 88.9%  |
| Jiang *et al.* [17] | Mesh | 90.5%  |
| *Ours* | 2D Convolution | 89.1%  |


## Attribution
The code for this experiment is modified from [UGSCNN](https://github.com/maxjiang93/ugscnn/tree/master/experiments/exp2_modelnet40). Our modifications are primarily to the model itself and to the dataloader, so that we can evaluate tangent images. The documentation below is from the original repository.

---------------------

### Dependencies
The code below has the following dependencies that can be installed by conda and pip.
```bash
conda install -c conda-forge rtree shapely  
conda install -c conda-forge pyembree  
pip install "trimesh[easy]"  
```
### Instruction
To run the experiment, execute the run script:
```bash
chmod +x run.sh
./run
```
The script will automatically start downloading the data files if it does not already exist.