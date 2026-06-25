#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_ID=0
SEEDS=(0 1 42)
BUDGET=500
OOD="50%"

FEDISIC_METHODS=("Random" "PAL" "LfOSA" "Entropy" "FEAL" "OpenPath")
FEDEMBED_METHODS=("Random" "PAL" "LfOSA" "Entropy" "FEAL" "OpenPath")

echo "=========================================================="
echo "RUNNING: Coldstart (no VLM)"
echo "=========================================================="

for SEED in "${SEEDS[@]}"; do
  for AL_METHOD in "${FEDISIC_METHODS[@]}"; do
    echo ">>> FedISIC | Seed ${SEED} | ${AL_METHOD}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
      --dataset FedISIC \
      --al_method ${AL_METHOD} \
      --budget ${BUDGET} \
      --max_round 15 \
      --al_round 5 \
      --mixed_precision \
      --base_lr 5e-4 \
      --seed ${SEED} \
      --deterministic \
      --ood ${OOD} \
      --explore_ratio 0.0 \
      --logs_folder "logs/FedISIC_FarOOD_v4_fixed" \
      --project_name "Coldstart5e-4_Round_15_Method_${AL_METHOD}"
    echo "----------------------------------------------------------"
  done
done

for SEED in "${SEEDS[@]}"; do
  for AL_METHOD in "${FEDEMBED_METHODS[@]}"; do
    echo ">>> FedEMBED | Seed ${SEED} | ${AL_METHOD}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
      --dataset FedEMBED \
      --al_method ${AL_METHOD} \
      --budget ${BUDGET} \
      --max_round 50 \
      --al_round 10 \
      --mixed_precision \
      --base_lr 3e-4 \
      --seed ${SEED} \
      --deterministic \
      --ood ${OOD} \
      --explore_ratio 0.0 \
      --logs_folder "logs/FedEMBED_v4_fixed" \
      --project_name "Coldstart3e-4_Round_50_Method_${AL_METHOD}_FedEMBED"
    echo "----------------------------------------------------------"
  done
done
