# Semantic Segmentation and Network Transfer

This directory has all the code for both the semantic segmentation and network transfer experiments. No additional dependencies are needed

## Datasets

To run these experiments, you will first need to download the Stanford 2D-3D-S dataset and/or the SYNTHIA and OmniSYNTHIA datasets.

Information for downloading the Stanford dataset [can be found here](http://buildingparser.stanford.edu/dataset.html#Download)

For SYNTHIA training, you need to download the `SYNTHIA-SEQS-01(02,04,05,06)-SUMMER` packages from the SYNTHIA VIDEO SEQUENCES subset. These [can be downloaded here](http://synthia-dataset.net/downloads/). OmniSYNTHIA is derived from this same subset. Code to generate OmniSYNTHIA images from the downloaded SYNTHIA data can be found here.

## Running this code

These experiments are designed to be run using the provided bash files with associated config files. Both `DataParallel` and `DistributedDataParallel` options are provided. We highly recommend the distributed versions, as they are much faster both both training and evaluation.


### Evaluating a model

Models are provided via [this Google Drive directory](https://drive.google.com/drive/folders/1d-agvJ55pi5Oo9Y91pBKvaNtmkdeQfz9?usp=sharing).

To run evaluation, use the command:

```
bash run_{dist_}test.sh <path/to/config.yaml>
```

### Training a model

To retrain a model according to the paper, simply specify one of the provided config files in the training command:

```
bash run_{dist_}train.sh <path/to/config.yaml>
```

It will automatically use all GPUs. To visualize training, start a visdom server in a `screen` or `tmux` environment before running the train script and set the appropriate flags in the config file.

### Config Files

Baseline config files for all experiments are located in the [configs/train](./configs/train) and [configs/eval](./configs/eval) directories.

Below is a table associating each config with an experiment from the updated arXiv version of paper. Here's some quick shorthand to understand:

`d###`: Trained on camera-normalized perspective images with a square dimension of ### pixels (e.g. `d128`)
`b#`: Base level of the tangent images used (e.g. `b1`)
`s#`: Input resolution level of spherical images in terms of icosahedral level (e.g. `s10`)

For any Stanford dataset experiment, the provided configs are preset to Fold 1 of the dataset. You will need to change the configs manually to run other folds.

#### Stanford Semantic Segmentation Experiments

##### Training

| ID | Config File | Experiment |
| --- | --- | --- |
| T0 | [`semseg-s5.yaml`](./configs/train/semseg-s5.yaml) | Train on (s5, b0) Stanford tangent images |
| T1 | [`semseg-s7.yaml`](./configs/train/semseg-s7.yaml) | Train on (s7, b0) Stanford tangent images |
| T2 | [`semseg-s10.yaml`](./configs/train/semseg-s10.yaml) | Train on (s10, b1) Stanford tangent images |


##### Eval

| ID | Config File | Experiment |
| --- | --- | --- |
| E0 | [`semseg-s5.yaml`](./configs/eval/semseg-s5.yaml) | Evaluate the model trained by T0 |
| E1 | [`semseg-s7.yaml`](./configs/eval/semseg-s7.yaml) | Evaluate the model trained by T1 |
| E2 | [`semseg-s10.yaml`](./configs/eval/semseg-s10.yaml) | Evaluate the model trained by T2 |


#### Stanford 2D-3D-S Transfer Experiments

##### Training

| ID | Config File | Experiment |
| --- | --- | --- |
| T3 | [`transfer-stanford-data-train.yaml`](./config/train/transfer-stanford-data-train.yaml) | Train on d128 Stanford perspective images |
| T4 | [`transfer-stanford-data-finetune.yaml`](./config/train/transfer-stanford-data-finetune.yaml) | Fine-tune the d128 network for 10 more epochs on d128 Stanford perspective images |
| T5 | [`transfer-stanford-pano-finetune.yaml`](./config/train/transfer-stanford-pano-finetune.yaml) | Fine-tune the d128 network for 10 epochs on (s8, b1) Stanford tangent images |


##### Eval

| ID | Config File | Experiment |
| --- | --- | --- |
| E3 | [`transfer-stanford-data.yaml` ](./config/eval/transfer-stanford-data.yaml`)| Evaluate a model trained by T3 or T4 |
| E4 | [`transfer-stanford-pano.yaml` ](./config/eval/transfer-stanford-pano.yaml`)| Evaluate a model trained by T3 or T5 |


#### (Omni)SYNTHIA Transfer Experiments

##### Training

| ID | Config File | Experiment |
| --- | --- | --- |
| T6 | [`transfer-synthia-data-dim32-train.yaml`](./configs/train/transfer-synthia-data-dim32-train.yaml) | Train on d32 SYNTHIA perspective images |
| T7 | [`transfer-synthia-data-dim64-train.yaml`](./configs/train/transfer-synthia-data-dim64-train.yaml) | Train on d64 SYNTHIA perspective images |
| T8 | [`transfer-synthia-data-dim128-train.yaml`](./configs/train/transfer-synthia-data-dim128-train.yaml) | Train on d128 SYNTHIA perspective images |
| T9 | [`transfer-synthia-s6-finetune.yaml`](./configs/train/transfer-synthia-s6-finetune.yaml`) | Fine-tune on (s6, b1) OmniSYNTHIA tangent images (initialize with model trained by T6) |
| T10 | [`transfer-synthia-s7-finetune.yaml`](./configs/train/transfer-synthia-s7-finetune.yaml`) | Fine-tune on (s7, b1) OmniSYNTHIA tangent images (initialize with model trained by T7) |
| T11 | [`transfer-synthia-s8-finetune.yaml`](./configs/train/transfer-synthia-s8-finetune.yaml`) | Fine-tune on (s8, b1) OmniSYNTHIA tangent images (initialize with model trained by T8) |


##### Eval

| ID | Config File | Experiment |
| --- | --- | --- |
| E5 | [`transfer-synthia-s6.yaml`](./configs/eval/transfer-synthia-s6.yaml) | Evaluate a model trained by T6 or T9 |
| E5 | [`transfer-synthia-s7.yaml`](./configs/eval/transfer-synthia-s7.yaml) | Evaluate a model trained by T7 or T10 |
| E6 | [`transfer-synthia-s8.yaml`](./configs/eval/transfer-synthia-s8.yaml) | Evaluate a model trained by T8 or T11 |


#### Config Options

All config file options are defined and documented in [ss/config_defaults.py](./ss/config_defaults.py).

