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
# How frequently to run validation (in epochs)
_C.VALIDATION_FREQ = 1
# How frequently to save prediction samples (-1 means never)
_C.SAMPLE_FREQ = -1
# Where to save checkpoints (default='./experiments')
_C.CHECKPOINT_ROOT = './experiments'
# Where to save any output samples (default='./samples')
_C.SAMPLE_ROOT = './samples'
# Where to save any output logs (default='./logs')
_C.LOGFILE = ''


# ####### Training settings
# Initial learning rate
_C.LR = 1e-3
# Start fresh (0), continue (-1), or start from specified epoch.
_C.START_EPOCH = 0
# Random sample size.
_C.RANDOM_SAMPLE_SIZE = 20
# Learning rate scheduler: 'step', 'multistep', 'thirdparty'.
_C.SCHEDULER = 'step'
# Label weight for the loss: 'synthia-none', 'synthia-ours', 'stanford-ours', 'stanford-thirdparty'.
_C.LABEL_WEIGHT = 'stanford-ours'
# Number of training epochs
_C.NUM_EPOCHS = 200
# Model path, if empty, will use the info from START_EPOCH
_C.MODEL_PATH = ''
# Setting this to true will only load the weights, not the optimization parameters, epoch info, etc. This is automatically set to True during eval
_C.LOAD_WEIGHTS_ONLY = False
# How frequently to save a checkpoint
_C.CHECKPOINT_FREQ = 1
# How frequently to visualize status
_C.VIZ_FREQ = 15
# Debug mode (1), standard(2), print status only(4), quiet(8)
_C.OP_MODE = 2

# ####### Tangent images settings
# parameters for 'pano' mode
_C.BASE_ORDER = 0
_C.SAMPLE_ORDER = 7
# parameter for 'data' mode.
_C.FOV = 45.0
_C.DIM = 512
# Eval on re-rendered equirectangular image ('rect') or on face of icosahedron ('ico')
_C.EVAL_FORMAT = 'rect'
# How to take the mean stats. 'weighted' means per-class frequency weighted summation, 'simple' means simple average
_C.MEAN_TYPE = 'simple'
# Run evaluation on the entire tangent image set at once (0) or on a per patch basis (>0, multiple of 4)
_C.EVAL_PER_PATCH = 0
# Whether to normalize intrinsics
_C.NORMALIZE_INTRINSICS = True

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
# Cache the dataset before running the model
_C.CACHE = False
# Cache root (if empty uses DATA_ROOT
_C.CACHE_ROOT = ''


# ####### GPU settings
# If empty list, uses all GPUs, otherwise list of GPU indices
_C.GPU_LIST = []


# ####### Data loader settings
_C.BATCH_SIZE_PER_GPU = 4
_C.NUM_WORKERS_PER_GPU = 4


# ####### Visualization
_C.VISDOM = CN()
_C.VISDOM.USE_VISDOM = True
_C.VISDOM.SERVER = 'localhost'
_C.VISDOM.DISPLAY_SAMPLES = True