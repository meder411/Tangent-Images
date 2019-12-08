config_file=${1}
ngpus=${2:-4}  # number of GPUs, 4 by default
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=${ngpus} main.py --config_file=${config_file} --distributed --evaluate ${@:3}
