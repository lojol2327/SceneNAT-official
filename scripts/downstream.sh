#!/bin/bash

# This script runs downstream task evaluations for the SceneNAT model.
#
# Usage:
#   1. Run ALL tasks for ALL rooms with 'best' checkpoint and 30 timesteps:
#      ./scripts/downstream.sh <model_version>
#      Example: ./scripts/downstream.sh 0723
#
#   2. Run a SPECIFIC task for ALL rooms:
#      ./scripts/downstream.sh <model_version> <task_name>
#      Example: ./scripts/downstream.sh 0723 completion
#
#   3. Run a specific task for a SPECIFIC room:
#      ./scripts/downstream.sh <model_version> <task_name> <room_type>
#      Example: ./scripts/downstream.sh 0723 completion bed
#
#   4. Run with a specific checkpoint:
#      ./scripts/downstream.sh <model_version> <task_name> <room_type> <checkpoint>
#      Example: ./scripts/downstream.sh 0723 completion bed 999
#
#   5. Run with specific timesteps:
#      ./scripts/downstream.sh <model_version> <task_name> <room_type> <checkpoint> <timesteps>
#      Example: ./scripts/downstream.sh 0723 completion bed 999 50
#
# Available tasks: completion, rearrangement, layouto, uncond, stylization

# --- Argument Parsing ---
MODEL_VERSION=$1
TASK_ARG=${2}
ROOM_TYPE_ARG=${3}
CHECKPOINT=${4:-best} # Default to 'best' if not provided
TIMESTEPS=${5:-30}    # Default to 30 if not provided

# --- Configuration ---
export CUDA_VISIBLE_DEVICES=0
TEMPERATURES=(0 1 2)
# --- End Configuration ---

# --- Validation and Setup ---
if [ -z "$MODEL_VERSION" ]; then
    echo "Error: Model version not provided."
    echo "Usage: $0 <model_version> [<task_name>] [<room_type>] [<checkpoint>] [<timesteps>]"
    exit 1
fi

ALL_TASKS=("completion" "rearrangement" "layouto" "uncond" "stylization")
if [ -n "$TASK_ARG" ]; then
    if [[ ! " ${ALL_TASKS[@]} " =~ " ${TASK_ARG} " ]]; then
        echo "Error: Invalid task name '${TASK_ARG}'."
        echo "Available tasks: ${ALL_TASKS[@]}"
        exit 1
    fi
    TASKS=("$TASK_ARG")
else
    TASKS=("${ALL_TASKS[@]}")
fi

if [ -n "$ROOM_TYPE_ARG" ]; then
    ROOM_TYPES=("$ROOM_TYPE_ARG")
else
    ROOM_TYPES=("bed" "living" "dining")
fi

export PYTHONPATH="$(pwd):${PYTHONPATH}"
BASE_OUTPUT_DIR="output/${MODEL_VERSION}"

# --- Main Execution Logic ---
echo "Starting downstream evaluation for model version: ${MODEL_VERSION}"
echo "Tasks to run: ${TASKS[@]}"
echo "Rooms to evaluate: ${ROOM_TYPES[@]}"
echo "Checkpoint setting: ${CHECKPOINT}"
echo "========================================"

# Construct checkpoint loading arguments
CHECKPOINT_ARG=""
if [ "$CHECKPOINT" = "best" ]; then
    CHECKPOINT_ARG="--use_best"
elif [ "$CHECKPOINT" != "latest" ]; then
    CHECKPOINT_ARG="--checkpoint_epoch ${CHECKPOINT}"
fi

for task in "${TASKS[@]}"; do
    for room_type in "${ROOM_TYPES[@]}"; do
        echo "--- Running Task: ${task}, Room: ${room_type} ---"

        # Base arguments common to most scripts
        COMMON_ARGS="--model_version ${MODEL_VERSION} --tag ${room_type}_46 --output_dir ${BASE_OUTPUT_DIR} --timesteps ${TIMESTEPS} --visualize ${CHECKPOINT_ARG}"

        case $task in
            completion|layouto)
                for temp in "${TEMPERATURES[@]}"; do
                    python -m "src.tasks.${task}" ${COMMON_ARGS} --visualize_partial --temperature ${temp}
                done
                ;;
            rearrangement)
                for temp in "${TEMPERATURES[@]}"; do
                    python -m "src.tasks.${task}" ${COMMON_ARGS} --visualize_messy --temperature ${temp}
                done
                ;;
            uncond)
                # Unconditional has its own temperature settings in the original script
                for temp in 7 8; do
                     python -m "src.tasks.${task}" ${COMMON_ARGS} --temperature ${temp}
                done
                ;;
            stylization)
                # Stylization doesn't loop over temperatures in the original script
                python -m "src.tasks.${task}" ${COMMON_ARGS} --visualize_original
                ;;
        esac
        
        if [ $? -ne 0 ]; then
            echo "Evaluation failed for task: ${task}, room: ${room_type}"
            exit 1
        fi
    done
done

echo "========================================"
echo "All downstream evaluations completed successfully."
