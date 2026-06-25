import argparse

# METHOD CONFIGURATIONS 
METHOD_CONFIGS = {
    'Random': {
        'description': 'Random sampling, no hyperparameters needed.'
    },
    'Entropy': {
        'description': 'Standard Uncertainty Sampling using Entropy.',
        'inference_batch_size': 64
    },
    'FEAL': {
        'description': 'Federated EDL Active Learning',
        'kl_weight': 0.01,
        'annealing_step': 10,
        'n_neighbor': 5,
        'cosine': 0.85
    },
    # Openset AL
    'PAL': {
        'description': 'Progressive Active Learning'
    },
    'LfOSA': {
        'description': 'Learning from Open Set Annotations (Discriminative)'
    },
    # Baseline Comparisons
    'OpenPath': {
        'description': 'OpenPath* Centroid-based ID filtering + KMeans++ diverse selection',
    }
}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General Experiment Settings 
    parser.add_argument('--project_name', type=str, default='FEAL_Openset')
    parser.add_argument('--dataset', type=str, default='FedISIC')
    parser.add_argument('--ood', type=str, default='ID', choices=['ID', '5%', '10%', '50%'])
    parser.add_argument('--fl_method', type=str, default='FedAvg')

    # Active Learning Strategy 
    parser.add_argument('--al_method', type=str, default='Random',
                        choices=METHOD_CONFIGS.keys(), help='Active Learning method')

    # --- Rounds & Budget ---
    parser.add_argument('--max_round', type=int, default=100, help='Total FL rounds per AL round')
    parser.add_argument('--al_round', type=int, default=5, help='Total AL rounds')
    parser.add_argument('--budget', type=int, default=500, help='Query budget per client')
    parser.add_argument('--query_ratio', type=float, default=0, help='Query ratio (overrides budget if > 0)')
    parser.add_argument('--query_model', type=str, default='local', choices=['global', 'local', 'both'])

    # --- Training ---
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_lr', type=float, default=5e-4)
    parser.add_argument('--mixed_precision', action='store_true', help='Use Mixed Precision (AMP) training')
    parser.add_argument('--save_model_weights', action='store_true', help='Save training models in each AL round')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--avoid_save_coop_vectors', action='store_true')
    parser.add_argument('--only_id_coop', action='store_true', help="Enable Dynamic VLM (Requires adapter)")

    parser.add_argument('--seed', type=int, default=0)

    # # --- VLM Settings ---
    parser.add_argument('--vlm_eval', action='store_true', help="Evaluate VLM Purity (Train) and Acc (Test) every round")
    parser.add_argument('--explore_ratio', type=float, default=0.0,
                        help="Ratio of budget (0.0-1.0) allocated to the Exploration Pool. "
                             "If 0.0, strict filtering is applied (only Safe Pool is queried).")

    parser.add_argument('--filter_strategy', type=str, default='vlm_only',
                        choices=['vlm_only', 'dot_product', 'scalar_product'],
                        help="Filtering Strategy (Round > 0):\n"
                             "- 'vlm_only': VLM Only. Uses VLM ID Probability > 0.5.\n"
                             "- 'dot_product': Strict Class Agreement. (VLM_Vector • Task_Vector).\n"
                             "- 'scalar_product': Soft Agreement. (VLM_Total_Prob * Task_Confidence).")
    # --- VLM Settings ---
    parser.add_argument('--warmup', type=str, default='random',
                        choices=['random', 'biomedclip', 'biomedclip_random',
                                 'biomedclip_highest', 'biomedclip_balanced', 'biomedclip_stratified'],
                        help="Strategy for Round 0 selection")

    parser.add_argument('--vlm_filter', action='store_true', help="Enable VLM filtering")
    parser.add_argument('--vlm_dynamic', action='store_true', help="Enable Dynamic VLM (Requires adapter)")

    parser.add_argument('--coop_shots', type=int, default=-1)
    parser.add_argument('--vlm_train_source', type=str, default='labeled',
                        choices=['query', 'labeled'],
                        help="Source data for VLM adapter training: "
                             "'query' = Use only samples selected in the current AL round (Few-Shot). "
                             "'labeled' = Use the full accumulated labeled set (Standard).")

    # ADAPTER TYPE
    parser.add_argument('--vlm_adapter', type=str, default=None,
                        choices=[None, 'CoOp_original', 'CoCoOp', 'ResCoOp'],
                        help="Adapter type. 'CoOp_original' = Standard. 'CoCoOp' = Conditional. 'ResCoOp' = Residual Federated.")
    parser.add_argument('--rescoop_orth_reg', action='store_true', default=False, help="Enable Orthogonal Regularization for ResCoOp to prevent dead residuals.")

    # FEDERATED SPLIT MODE
    parser.add_argument('--coop_federated', action='store_true',
                        help="If True, splits vectors into Shared (Global) and Private (Local).")
    # VECTOR COUNTS (Applied if coop_federated is True)
    parser.add_argument('--coop_global_vectors', type=int, default=8, help="Number of Global vectors")
    parser.add_argument('--coop_local_vectors', type=int, default=8, help="Number of Local vectors")
    # FUSION STRATEGY (Applied if coop_federated is True)
    parser.add_argument('--vlm_fusion_strategy', type=str, default='concat',
                        choices=['concat', 'ensemble'],
                        help="How to combine Global/Local. 'concat' = Feature Fusion. 'ensemble' = Logit Ensemble.")

    parser.add_argument('--vlm_ensemble_alpha', type=float, default=0.5,
                        help="Weight for Global branch in ensemble mode (0.0 to 1.0).")
    # Standard CoOp Args
    parser.add_argument('--coop_vectors', type=int, default=16, help="Total vectors (used if NOT federated split)")
    parser.add_argument('--coop_epochs', type=int, default=20)

    #  Logging 
    parser.add_argument('--display_freq', type=int, default=100, help='Frequency of testing/display')
    parser.add_argument('--logs_folder', type=str, default="logs", help='Folder name to save logs')

    parser.add_argument('--vlm_csc', action='store_true',
                    help="Enable Class-Specific Context (CSC). If False, uses Unified Context.")

    # --- OpenPath* Baseline ---
    parser.add_argument('--openpath_id_ratio', type=float, default=0.25,
        help="Fraction of unlabeled pool to keep as ID candidates via centroid filtering (OpenPath)")

    return parser.parse_args()
