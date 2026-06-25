import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, args, method_config):
        self.args = args
        self.method_config = method_config

        # 1. SETUP DIRECTORY STRUCTURE
        ticks = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = os.path.join(
            args.logs_folder,
            args.dataset,
            args.al_method,
            f"seed_{args.seed}",
            ticks
        )
        self.model_dir = os.path.join(self.base_dir, 'models')

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # 2. GLOBAL LOGGING SETUP
        logging.basicConfig(
            filename=os.path.join(self.base_dir, "experiment_log.txt"),
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.console = logging.StreamHandler(sys.stdout)
        logging.getLogger().addHandler(self.console)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.base_dir, 'tb_logs'))

        # 3. GLOBAL EXPERIMENT SUMMARY CSV
        self.global_csv_path = os.path.join(self.base_dir, "global_experiment_summary.csv")

        self.global_headers = [
            "AL_Round",
            "Avg_Test_Acc", "Avg_Bal_Acc", "Avg_Precision", "Avg_Recall", "Avg_F1", "Avg_AUC",
            "Avg_QR", "Avg_AQR",
            "Total_Labeled_Samples", "Total_Queries", "ID_Purity",
            "Round_Train_Time_Sec", "Total_Elapsed_Time_Sec"
        ]
        if not os.path.exists(self.global_csv_path):
            pd.DataFrame(columns=self.global_headers).to_csv(self.global_csv_path, index=False)
        else:
            # Normalize headers (e.g. after manual CSV edits that add whitespace)
            existing = pd.read_csv(self.global_csv_path)
            existing.columns = existing.columns.str.strip()
            existing.to_csv(self.global_csv_path, index=False)

        self.log_config()

    def log_config(self):
        full_config = vars(self.args).copy()
        full_config.update(self.method_config)
        config_serializable = {k: str(v) for k, v in full_config.items()}
        config_str = json.dumps(config_serializable, indent=4)
        logging.info(f"Configuration:\n{config_str}")
        print(f"Experiment saved to: {self.base_dir}")

    def create_al_round_directory(self, al_round):
        self.current_al_dir = os.path.join(self.base_dir, f"AL_Round_{al_round}")
        os.makedirs(self.current_al_dir, exist_ok=True)

        # A. Test Metrics CSV (Updated)
        self.test_csv_path = os.path.join(self.current_al_dir, "metrics_test.csv")

        cls_headers = []
        n_classes = getattr(self.args, 'num_classes', 8)
        for i in range(n_classes):
            cls_headers.extend([f"Class_{i}_Count", f"Class_{i}_Prec", f"Class_{i}_Rec", f"Class_{i}_F1"])

        self.test_headers = ["FL_Round", "Client_ID", "Accuracy", "Balanced_Acc", "Precision", "Recall", "F1_Score", "AUC", "QR", "AQR"] + cls_headers
        pd.DataFrame(columns=self.test_headers).to_csv(self.test_csv_path, index=False)

        # B. Train Metrics CSV
        self.train_csv_path = os.path.join(self.current_al_dir, "metrics_train.csv")
        self.train_headers = ["FL_Round", "Client_ID", "Loss"]
        pd.DataFrame(columns=self.train_headers).to_csv(self.train_csv_path, index=False)

        # C. Client Data Stats CSV
        self.stats_csv_path = os.path.join(self.current_al_dir, "client_data_stats.csv")
        n_id = getattr(self.args, 'num_classes', 8)
        n_ood = getattr(self.args, 'num_ood_classes', 1)
        total_log_cols = n_id + n_ood
        cls_cols = [f"Class_{i}" for i in range(total_log_cols)]
        self.stats_headers = ["Client_ID", "Set_Type", "Total_Count", "Score_Mean", "Score_Std", "Purity"] + cls_cols
        pd.DataFrame(columns=self.stats_headers).to_csv(self.stats_csv_path, index=False)

        # D. VLM Metrics CSV (Updated - Removed Global/Total)
        self.vlm_csv_path = os.path.join(self.current_al_dir, "metrics_vlm_breakdown.csv")

        vlm_cls_headers = []
        sets = ["Unlabeled", "Pool_ID", "Pool_Explore", "Query", "Labeled"]
        for s in sets:
            for i in range(n_classes):
                vlm_cls_headers.extend([f"{s}_Count_Class_{i}", f"{s}_Prec_Class_{i}", f"{s}_Rec_Class_{i}", f"{s}_F1_Class_{i}"])
        self.vlm_headers = [
            "FL_Round", "Client_ID",
            "Unlabeled_VLM_Acc_OOD", "Unlabeled_ID_Recall", "Unlabeled_Dist",
            "Unlabeled_Count", "Unlabeled_Purity", "Unlabeled_VLM_Acc_ID",
            "Pool_ID_Count", "Pool_ID_Purity", "Pool_ID_VLM_Acc_ID", "Pool_ID_Dist", "Pool_ID_ID_Recall", "Pool_ID_F1",
            "Pool_Explore_Count", "Pool_Explore_Purity", "Pool_Explore_VLM_Acc_ID", "Pool_Explore_Dist", "Pool_Explore_ID_Recall", "Pool_Explore_F1",
            "Query_Count", "Query_Purity", "Query_VLM_Acc_ID", "Query_Dist", "Query_ID_Recall", "Query_F1",
            "Labeled_Count", "Labeled_Purity", "Labeled_VLM_Acc_ID", "Labeled_Dist", "Labeled_ID_Recall", "Labeled_F1"
        ] + vlm_cls_headers

        pd.DataFrame(columns=self.vlm_headers).to_csv(self.vlm_csv_path, index=False)

    def log_test_metrics(self, fl_round, client_id, metrics_dict):
        row_data = {"FL_Round": fl_round, "Client_ID": client_id, **metrics_dict}
        row_ordered = {k: row_data.get(k, 0.0) for k in self.test_headers}
        df = pd.DataFrame([row_ordered])
        df.to_csv(self.test_csv_path, mode='a', header=False, index=False)

        for key, val in metrics_dict.items():
            tag = f"Client_{client_id}/{key}" if client_id != 'Global' else f"Global/{key}"
            current_al_idx = getattr(self.args, 'current_al_round', 0)
            step = (current_al_idx * self.args.max_round) + fl_round
            self.writer.add_scalar(tag, val, step)

    def log_vlm_metrics(self, fl_round, client_id, metrics_dict):
        row_data = {"FL_Round": fl_round, "Client_ID": client_id, **metrics_dict}
        row_ordered = {k: row_data.get(k, '') for k in self.vlm_headers}
        df = pd.DataFrame([row_ordered])
        df.to_csv(self.vlm_csv_path, mode='a', header=False, index=False)

    def log_train_metric(self, fl_round, client_id, loss_val):
        row = [fl_round, client_id, loss_val]
        df = pd.DataFrame([row], columns=self.train_headers)
        df.to_csv(self.train_csv_path, mode='a', header=False, index=False)

        current_al_idx = getattr(self.args, 'current_al_round', 0)
        step = (current_al_idx * self.args.max_round) + fl_round
        tag = f"Client_{client_id}/Train_Loss"
        self.writer.add_scalar(tag, loss_val, step)

    def log_client_state(self, client_id, set_type, counts_dict, scores=None, purity=0.0, true_total=None):
        if scores is not None and len(scores) > 0:
            s_mean = np.mean(scores)
            s_std = np.std(scores)
        else:
            s_mean = 0.0
            s_std = 0.0

        total = true_total if true_total is not None else sum(counts_dict.values())

        row_dict = {
            "Client_ID": client_id,
            "Set_Type": set_type,
            "Total_Count": total,
            "Score_Mean": s_mean,
            "Score_Std": s_std,
            "Purity": purity
        }

        n_id = getattr(self.args, 'num_classes', 8)
        n_ood = getattr(self.args, 'num_ood_classes', 1)
        total_cols = n_id + n_ood

        for i in range(total_cols):
            row_dict[f"Class_{i}"] = counts_dict.get(i, 0)

        df = pd.DataFrame([row_dict])
        df = df[self.stats_headers]
        df.to_csv(self.stats_csv_path, mode='a', header=False, index=False)

    def log_selected_samples(self, client_id, selected_indices, original_dataset, scores=None):
        data_rows = []
        for rank, idx in enumerate(selected_indices):
            try:
                _, sample_data = original_dataset[idx]
                path = sample_data.get('path', 'N/A')
                orig_label = sample_data.get('original_label', -1)
                if hasattr(orig_label, 'item'):
                    orig_label = orig_label.item()
            except Exception:
                path = "Error"
                orig_label = -1

            score = scores[rank] if scores is not None else 0.0

            data_rows.append({
                "Rank": rank,
                "Dataset_Index": idx,
                "Path": path,
                "Original_Label": orig_label,
                "Score": score
            })

        df = pd.DataFrame(data_rows)
        save_path = os.path.join(self.current_al_dir, f"query_selected_client{client_id}.csv")
        df.to_csv(save_path, index=False)

    def log_global_summary(self, al_round, metrics_dict, total_labeled, total_queries, id_purity=0.0, avg_qr=0.0, avg_aqr=0.0, round_train_time=0.0, total_elapsed_time=0.0):
        row = [
            al_round,
            metrics_dict.get('Accuracy', 0.0),
            metrics_dict.get('Balanced_Acc', 0.0),
            metrics_dict.get('Precision', 0.0),
            metrics_dict.get('Recall', 0.0),
            metrics_dict.get('F1_Score', 0.0),
            metrics_dict.get('AUC', 0.0),
            avg_qr,
            avg_aqr,
            total_labeled,
            total_queries,
            id_purity,
            round_train_time,
            total_elapsed_time
        ]
        df = pd.DataFrame([row], columns=self.global_headers)
        df.to_csv(self.global_csv_path, mode='a', header=False, index=False)

    def close(self):
        self.writer.close()
