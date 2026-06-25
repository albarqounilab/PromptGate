# `training/` — Training Loops

Contains the outer Active Learning loop and the inner Federated Learning loop.

## Glossary

| Abbreviation | Expansion |
|---|---|
| AL   | Active Learning — iterative querying of the most informative unlabeled samples |
| FL   | Federated Learning — collaborative training across clients without sharing raw data |
| OOD  | Out-of-Distribution |
| ID   | In-Distribution |
| QP   | Query Precision — fraction of queried samples that are ID |
| AQR  | Average Query Recall — cumulative fraction of all ID samples discovered |
| VLM  | Vision-Language Model (BiomedCLIP) |
| CoOp | Context Optimization — prompt-tuning adapter for CLIP |
| OVA  | One-vs-All (binary classifiers used by PAL) |
| WNet | Weighting Network — meta-learning module used by PAL |
| AMP  | Automatic Mixed Precision (`torch.amp`) |

## Files

| File | Purpose |
|------|---------|
| `active_learning_loop.py` | `run_active_learning_loop()` — the outer loop. Iterates over AL rounds: re-initialises the model, runs FL training, queries new samples, updates the VLM adapter, and aggregates global prompt vectors. |
| `federated_loop.py` | `run_federated_loop()` — the inner loop. Runs `args.max_round` FL rounds of local training + `fed_avg` aggregation + periodic evaluation. Also exposes `prepare_ood_loaders()`, `build_client_optimizers()`, and `reinitialize_or_warmstart_model()`. |

## Loop Structure

```
for al_round in AL Rounds:                        ← active_learning_loop.py
    run_federated_loop()                          ← federated_loop.py
        for fl_round in FL Rounds:
            local training (all clients)
            fed_avg aggregation
            [periodic evaluation]
            fed_update back to local models
    query_samples()                               ← utils/cls/selection_methods.py
    [train_vlm_adapter()]                         ← utils/vlm_filter.py
    [fed_avg() for VLM global ctx vectors]
```

## Data Flow

```
local_data['train'][c]       → training dataset for client c
local_data['unlabeled'][c]   → pool dataset for client c (same images, different transforms)
local_sets['labeled'][c]     → List[int] — index subset used for training
local_sets['unlabeled'][c]   → List[int] — remaining unlabeled indices
local_sets['discarded'][c]   → List[int] — OOD-flagged indices (not used for ID training)
```
