#!/bin/bash

# This script runs evaluation for the SceneNAT model.
#
# Usage:
#   1. Evaluate with 'best' checkpoint for all rooms:
#      ./scripts/eval.sh <model_version>
#      Example: ./scripts/eval.sh 0903v1
#
#   2. Evaluate with a specific checkpoint for all rooms:
#      ./scripts/eval.sh <model_version> <checkpoint>
#      Example: ./scripts/eval.sh 0903v1 999
#      Example: ./scripts/eval.sh 0903v1 latest
#
#   3. Evaluate with a specific checkpoint and for a single room:
#      ./scripts/eval.sh <model_version> <checkpoint> <room_type>
#      Example: ./scripts/eval.sh 0903v1 best bed

# --- Argument Parsing ---
MODEL_VERSION=$1
CHECKPOINT=${2:-best} # Default to 'best' if not provided
ROOM_TYPE_ARG=$3

# --- Configuration ---
# GPU to use
export CUDA_VISIBLE_DEVICES=1
# Temperatures for sampling. Add more values if needed.
TEMPERATURES=(0)
# Timesteps for sampling
TIMESTEPS=50
# --- End Configuration ---

# Check if model version is provided
if [ -z "$MODEL_VERSION" ]; then
    echo "Error: Model version not provided."
    echo "Usage: $0 <model_version> [<checkpoint>] [<room_type>]"
    exit 1
fi

# Determine which room types to evaluate
if [ -n "$ROOM_TYPE_ARG" ]; then
    ROOM_TYPES=("$ROOM_TYPE_ARG")
else
    ROOM_TYPES=("bed" "living" "dining")
fi

export PYTHONPATH="$(pwd):${PYTHONPATH}"
BASE_OUTPUT_DIR="output/${MODEL_VERSION}"

echo "Starting evaluation for model version: ${MODEL_VERSION}"
echo "Checkpoint setting: ${CHECKPOINT}"
echo "Output directory will be based in: ${BASE_OUTPUT_DIR}"
echo "Rooms to evaluate: ${ROOM_TYPES[@]}"
echo "========================================"

for room_type in "${ROOM_TYPES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        echo "--- Running for Room: ${room_type}, Temperature: ${temp} ---"

        # Construct checkpoint loading arguments
        CHECKPOINT_ARG=""
        if [ "$CHECKPOINT" = "best" ]; then
            CHECKPOINT_ARG="--use_best"
        elif [ "$CHECKPOINT" != "latest" ]; then
            # If it's a number (or anything other than 'latest'/'best')
            CHECKPOINT_ARG="--checkpoint_epoch ${CHECKPOINT}"
        fi
        # If CHECKPOINT is 'latest', no argument is needed as it's the default behavior of the script

        python -m src.tasks.eval \
            --model_version ${MODEL_VERSION} \
            --output_dir ${BASE_OUTPUT_DIR} \
            --tag ${room_type} \
            --given_lengths \
            --timesteps ${TIMESTEPS} \
            --temperature ${temp} \
            --visualize \
            ${CHECKPOINT_ARG}
        
        if [ $? -ne 0 ]; then
            echo "Evaluation failed for room: ${room_type}, temp: ${temp}"
            exit 1
        fi

        echo "--- Finished for Room: ${room_type}, Temperature: ${temp} ---"
    done
done

echo "========================================"
echo "All evaluations completed successfully."
