#!/bin/bash

export PYTHONPATH="${workspaceFolder}:${env:PYTHONPATH}"

# GPU 설정
export CUDA_VISIBLE_DEVICES=3

model_version="triplet"
room_types=("living")
# room_types=("bed" "living" "dining")
sampling_steps=(50)
# (1599) #for room_types=("bed")
# (1999) #for room_types=("living")
# (1549) #for room_types=("dining")

# 각 temperature 값에 대해 실험 실행
for room_type in "${room_types[@]}"; do
    for step_num in "${sampling_steps[@]}"; do
        echo "Running evaluation on $model_version/$room_type with sampling steps: $step_num"
        
        python3 src/tasks/eval.py \
            --tag ${room_type} \
            --output_dir output/otrans3_scene_reweight \
            --given_lengths \
            --timesteps $step_num \
            --model_version $model_version \
            --checkpoint_epoch 1599 \
            --visualize \
            --eight_views \
            --resolution 1024 \
            # --all_steps \


        echo "Completed evaluation on $model_version/$room_type with sampling steps: $step_num"
        echo "----------------------------------------"
    done
done
    