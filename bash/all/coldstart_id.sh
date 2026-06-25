#!/bin/bash
# Coldstart with no OOD injection (--ood ID).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_ID=0
DATASETS=("FedEMBED")
AL_METHODS=("Random" "PAL" "LfOSA" "Entropy" "FEAL" "OpenPath")
BUDGET=500
OOD="ID"
SEEDS=(0 1 42)
LR_LIST=("3e-4")

echo "=========================================================="
echo "RUNNING: Coldstart ID (No OOD)"
echo "=========================================================="

for DATASET in "${DATASETS[@]}"; do
  if [ "$DATASET" == "FedISIC" ]; then
    AL_ROUND=5
    MAX_ROUND=15
    LR="5e-4"
  else
    AL_ROUND=10
    MAX_ROUND=50
    LR="3e-4"
  fi

  for SEED in "${SEEDS[@]}"; do
    for AL_METHOD in "${AL_METHODS[@]}"; do
      echo ">>> ${DATASET} | Seed ${SEED} | ${AL_METHOD} | max_round=${MAX_ROUND}"

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
        --explore_ratio 0.0 \
        --logs_folder "logs/${DATASET}_v4_fixed" \
        --project_name "Coldstart_ID_${LR}_Round_${MAX_ROUND}_Method_${AL_METHOD}_${DATASET}"

      echo "----------------------------------------------------------"
    done
  done
done
