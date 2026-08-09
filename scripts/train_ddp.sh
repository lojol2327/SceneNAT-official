#!/bin/bash

roomtype=bed
exp_name=bins_32
model="triplet"

export CUDA_VISIBLE_DEVICES=2
export MASTER_PORT=29509 # 29500 ~ 29999
export PYTHONPATH="${workspaceFolder}:${env:PYTHONPATH}"


NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export WORLD_SIZE=$NUM_GPUS
export NODE_RANK=0
export MASTER_ADDR=localhost

NPROC_PER_NODE=$NUM_GPUS
NUM_WORKERS=$((4 * NUM_GPUS))

python3 -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    src/tasks/train.py \
    --config_file configs/${roomtype}.yaml \
    --output_dir "output/" \
    --num_workers $NUM_WORKERS \
    --test \
    --tag ${exp_name}/${roomtype} \
    --model_version ${model} \
    --timesteps 50 \
    --resume