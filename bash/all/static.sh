#!/bin/bash
# Static VLM filter (Round-0 mask frozen). Matches logs/FedISIC_FarOOD_v4_fixed static runs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_ID=1
BUDGET=500
OOD="50%"
SEEDS=(0 1 42)

run_dataset() {
  local DATASET=$1
  local AL_METHODS=("${!2}")
  local MAX_ROUND=$3
  local AL_ROUND=$4
  local LR=$5
  local LOGS_FOLDER=$6

  for SEED in "${SEEDS[@]}"; do
    for AL_METHOD in "${AL_METHODS[@]}"; do
      echo ">>> ${DATASET} | Seed ${SEED} | ${AL_METHOD} | static VLM"
      CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --dataset ${DATASET} \
        --al_method ${AL_METHOD} \
        --budget ${BUDGET} \
        --max_round ${MAX_ROUND} \
        --al_round ${AL_ROUND} \
        --mixed_precision \
        --base_lr ${LR} \
        --seed ${SEED} \
        --deterministic \
        --ood ${OOD} \
        --warmup biomedclip_random \
        --filter_strategy vlm_only \
        --vlm_filter \
        --vlm_eval \
        --explore_ratio 0.0 \
        --logs_folder "${LOGS_FOLDER}" \
        --project_name "Static_Round_${MAX_ROUND}_Method_${AL_METHOD}_${DATASET}"
      echo "----------------------------------------------------------"
    done
  done
}

FEDISIC_METHODS=("Random" "PAL" "LfOSA" "Entropy" "FEAL" "OpenPath")
FEDEMBED_METHODS=("Random" "PAL" "LfOSA" "Entropy" "FEAL" "OpenPath")


echo "=========================================================="
echo "RUNNING: Static VLM"
echo "=========================================================="

run_dataset FedISIC FEDISIC_METHODS[@] 15 5 5e-4 "logs/FedISIC_FarOOD_v4_fixed"
run_dataset FedEMBED FEDEMBED_METHODS[@] 50 10 3e-4 "logs/FedEMBED_v4_fixed"
