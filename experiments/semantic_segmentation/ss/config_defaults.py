from yacs.config import CfgNode as CN

_C = CN()


# ####### Experiment settings
_C.DEVICE = 'cuda'
# Force experiment name
_C.EXPERIMENT_NAME = ''
# Suffix to append to generated experiment name
_C.EXPERIMENT_SUFFIX = ''
# Data format: 'data', 'pano'.
_C.DATA_FORMAT = 'pano'

_C.SAMPLE_FREQ = -1


# ####### Training settings
# Initial learning rate
_C.LR = 1e-3
# Start fresh (0), continue (-1), or start from specified epoch.
_C.START_EPOCH = 0
# Random sample size.
_C.RANDOM_SAMPLE_SIZE = 10
# Learning rate scheduler: 'step', 'multistep', 'thirdparty'.
_C.SCHEDULER = 'step'
# Label weight for the loss: 'synthia-none', 'synthia-ours', 'stanford-ours', 'stanford-thirdparty'.
_C.LABEL_WEIGHT = 'stanford-ours'
_C.NUM_EPOCHS = 200


# ####### Tangent plane settings
# parameters for 'pano' mode
_C.BASE_ORDER = 0
_C.SAMPLE_ORDER = 7
# parameter for 'data' mode.
_C.FOV = 45
_C.DIM = 512


# ####### Network settings
# Model type: 'zhangunet', 'resnet101', 'hexunet'.
_C.MODEL_TYPE = 'zhangunet'
# Apply nonlinearity after the first layer.
_C.INPUT_NONLIN = False
# Type of of input normalization: 'imagenet', 'custom', 'ugscnn'
_C.NORMALIZATION_STATS = 'custom'
# Drop unknown class (0) from the prediction set.
_C.DROP_UNKNOWN = False


# ####### Data settings
_C.FOLD = 0
# Data root folder.
_C.DATA_ROOT = './data/2D-3D-Semantics/'
# Dataset type: 'synthia', 'stanford'.
_C.DATASET = 'stanford'
# Use depth as the 4th input channel.
_C.USE_DEPTH = False

# ####### Data loader settings
_C.BATCH_SIZE_PER_GPU = 4
_C.NUM_WORKERS_PER_GPU = 4


# ####### Visualization
_C.VISDOM = CN()
_C.VISDOM.SERVER = 'localhost'
