#!/bin/bash
# Dynamic VLM — global-only CoOp vectors (G16 L0).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_ID=1
BUDGET=500
OOD="50%"
ADAPTER="CoOp_original"
SEEDS=(0 1 42)
FUSION="concat"
GLOBAL_VEC=16
LOCAL_VEC=0
COOP_EPOCHS=50

run_dataset() {
  local DATASET=$1
  local AL_METHODS=("${!2}")
  local MAX_ROUND=$3
  local AL_ROUND=$4
  local LR=$5
  local LOGS_FOLDER=$6

  for SEED in "${SEEDS[@]}"; do
    for AL_METHOD in "${AL_METHODS[@]}"; do
      echo ">>> ${DATASET} | Seed ${SEED} | ${AL_METHOD} | global-only CoOp"
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
        --vlm_dynamic \
        --vlm_adapter ${ADAPTER} \
        --coop_federated \
        --coop_shots 128 \
        --vlm_train_source labeled \
        --coop_global_vectors ${GLOBAL_VEC} \
        --coop_local_vectors ${LOCAL_VEC} \
        --vlm_fusion_strategy ${FUSION} \
        --coop_epochs ${COOP_EPOCHS} \
        --vlm_csc \
        --vlm_eval \
        --explore_ratio 0.0 \
        --logs_folder "${LOGS_FOLDER}" \
        --project_name "CSC_Ablation_${DATASET}_R${MAX_ROUND}_G${GLOBAL_VEC}_L${LOCAL_VEC}_${AL_METHOD}"
      echo "----------------------------------------------------------"
    done
  done
}

#FEDISIC_METHODS=("Random" "PAL" "LfOSA" "Entropy" "FEAL" "OpenPath")
FEDEMBED_METHODS=("Random" "PAL" "LfOSA" "Entropy" "FEAL" "OpenPath")

echo "=========================================================="
echo "RUNNING: Dynamic VLM — Global-only CoOp"
echo "=========================================================="

run_dataset FedISIC FEDISIC_METHODS[@] 15 5 5e-4 "logs/FedISIC_FarOOD_v4_fixed"
run_dataset FedEMBED FEDEMBED_METHODS[@] 50 10 3e-4 "logs/FedEMBED_v4_fixed"
