config_file=${1}
ngpus=${2:-4}  # number of GPUs, 4 by default
python -m torch.distributed.launch --nproc_per_node=${ngpus} main.py --config_file=${config_file} --distributed ${@:3}
