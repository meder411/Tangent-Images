## 3D Shape Classification on ModelNet40

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