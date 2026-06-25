import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentVisualizer:
    def __init__(self, base_dir, al_method):
        self.base_dir = base_dir
        self.al_method = al_method
        self.plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        # Set professional style
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams.update({'figure.max_open_warning': 0})

    def _load_global_summary(self):
        """Helper to safely load the global summary CSV."""
        csv_path = os.path.join(self.base_dir, "global_experiment_summary.csv")
        if not os.path.exists(csv_path): return None
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df

    def _load_client_stats(self):
        """Helper to load and aggregate client stats across rounds."""
        client_data = []
        al_dirs = sorted(glob.glob(os.path.join(self.base_dir, "AL_Round_*")))
        
        for d in al_dirs:
            try: r_num = int(d.split('_')[-1])
            except: continue
            
            stats_csv = os.path.join(d, "client_data_stats.csv")
            if not os.path.exists(stats_csv): continue
            
            df = pd.read_csv(stats_csv)
            # Filter for Query set
            query_df = df[df['Set_Type'] == 'Query'].copy()
            if not query_df.empty:
                query_df['AL_Round'] = r_num
                client_data.append(query_df)
                
        if not client_data: return None
        full_df = pd.concat(client_data)
        # Ensure Client_ID is string for categorical plotting
        full_df['Client_ID'] = full_df['Client_ID'].astype(str)
        return full_df

    def _load_client_metrics(self):
        """Helper to load client test metrics."""
        client_data = []
        al_dirs = sorted(glob.glob(os.path.join(self.base_dir, "AL_Round_*")))
        for d in al_dirs:
            try: r_num = int(d.split('_')[-1])
            except: continue
            test_csv = os.path.join(d, "metrics_test.csv")
            if not os.path.exists(test_csv): continue
            
            df = pd.read_csv(test_csv)
            if df.empty: continue
            
            # Get final FL round for this AL round
            max_fl = df['FL_Round'].max()
            final_df = df[df['FL_Round'] == max_fl].copy()
            
            clients_only = final_df[final_df['Client_ID'] != 'Global'].copy()
            clients_only['AL_Round'] = r_num
            client_data.append(clients_only)

        if not client_data: return None
        full_df = pd.concat(client_data)
        full_df['Client_ID'] = full_df['Client_ID'].astype(str)
        
        # Standardize column name if needed
        if 'Balanced_Acc' not in full_df.columns and 'Balanced_Accuracy' in full_df.columns:
            full_df.rename(columns={'Balanced_Accuracy': 'Balanced_Acc'}, inplace=True)
            
        return full_df

    # --- PLOT 1: Standard Learning Curve (Accuracy) ---
    def plot_learning_curve_detailed(self):
        df = self._load_global_summary()
        if df is None or df.empty: return

        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df, x='AL_Round', y='Avg_Test_Acc', 
                          marker='o', linewidth=3, markersize=10, color='tab:blue')
        
        for i in range(df.shape[0]):
            round_idx = df.iloc[i]['AL_Round']
            acc = df.iloc[i]['Avg_Test_Acc']
            samples = int(df.iloc[i]['Total_Labeled_Samples'])
            ax.text(round_idx, acc + 0.005, f"N={samples}", 
                    color='black', fontsize=10, ha='center', weight='bold')

        plt.title(f"Global Accuracy vs AL Rounds ({self.al_method})")
        plt.xlabel("Active Learning Round")
        plt.ylabel("Global Test Accuracy")
        plt.xticks(df['AL_Round'].unique())
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "1_Global_Learning_Curve.png"), bbox_inches='tight')
        plt.close()

    # --- PLOT 2: All Metrics Evolution ---
    def plot_all_global_metrics(self):
        df = self._load_global_summary()
        if df is None or df.empty: return
        metrics = ['Avg_Bal_Acc', 'Avg_Precision', 'Avg_Recall', 'Avg_F1', 'Avg_AUC']
        df_melt = df.melt(id_vars=['AL_Round'], value_vars=metrics, var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_melt, x='AL_Round', y='Score', hue='Metric', markers=True, style='Metric')
        plt.title("Evolution of All Global Metrics")
        plt.xticks(df['AL_Round'].unique())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "2_Global_Metrics_Evolution.png"), bbox_inches='tight')
        plt.close()

    # --- PLOT 4: Absolute Purity (0-1 Scale) ---
    def plot_purity_evolution(self):
        global_df = self._load_global_summary()
        client_df = self._load_client_stats()
        
        plt.figure(figsize=(10, 6))
        
        # Global Line
        if global_df is not None and 'ID_Purity' in global_df.columns:
            sns.lineplot(data=global_df, x='AL_Round', y='ID_Purity', 
                         marker='o', linewidth=3, color='black', label='Global Average')

        # Client Lines
        if client_df is not None and 'Purity' in client_df.columns:
            sns.lineplot(data=client_df, x='AL_Round', y='Purity', hue='Client_ID',
                         palette='tab10', linestyle='--', marker='X', alpha=0.7)

        plt.title(f"OOD Robustness: ID Purity (Absolute Scale)")
        plt.ylabel("ID Purity (1.0 = All ID)")
        plt.xlabel("Active Learning Round")
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "4_Purity_Evolution_Abs.png"), bbox_inches='tight')
        plt.close()

    # --- PLOT 7: Global Purity (Relative Scale) ---
    def plot_global_purity_relative(self):
        df = self._load_global_summary()
        if df is None or 'ID_Purity' not in df.columns: return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='AL_Round', y='ID_Purity', marker='o', linewidth=3, color='black', label='Global')

        # Annotate
        for i in range(df.shape[0]):
            val = df.iloc[i]['ID_Purity']
            r = df.iloc[i]['AL_Round']
            plt.text(r, val, f"{val:.1%}", fontsize=10, ha='center', va='bottom', weight='bold')

        plt.title(f"Global Purity (Relative Scale)")
        plt.xlabel("AL Round")
        plt.ylabel("ID Purity (Zoomed)")
        plt.xticks(df['AL_Round'].unique())
        
        # Dynamic Zoom
        y_min, y_max = df['ID_Purity'].min(), df['ID_Purity'].max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
        plt.ylim(y_min - margin, y_max + margin)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "7_Global_Relative_Purity.png"), bbox_inches='tight')
        plt.close()

    # --- PLOT 8: Client Purity (Relative Scale) ---
    def plot_client_purity_relative(self):
        df = self._load_client_stats()
        if df is None or 'Purity' not in df.columns: return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='AL_Round', y='Purity', hue='Client_ID', 
                     palette='tab10', marker='o', linewidth=2.5, linestyle='--')

        plt.title(f"Client Purity (Relative Scale)")
        plt.xlabel("AL Round")
        plt.ylabel("ID Purity (Zoomed)")
        plt.xticks(df['AL_Round'].unique())
        plt.legend(title="Client ID", bbox_to_anchor=(1.05, 1), loc='upper left')

        y_min, y_max = df['Purity'].min(), df['Purity'].max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
        plt.ylim(y_min - margin, y_max + margin)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "8_Client_Relative_Purity.png"), bbox_inches='tight')
        plt.close()

    # --- PLOT 9: Global Balanced Accuracy ---
    def plot_global_balanced_acc(self):
        df = self._load_global_summary()
        if df is None or 'Avg_Bal_Acc' not in df.columns: return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='AL_Round', y='Avg_Bal_Acc', marker='o', linewidth=3, color='tab:blue', label='Global')

        for i in range(df.shape[0]):
            acc = df.iloc[i]['Avg_Bal_Acc']
            r = df.iloc[i]['AL_Round']
            n_samples = int(df.iloc[i]['Total_Labeled_Samples'])
            plt.text(r, acc, f"N={n_samples}", fontsize=10, ha='center', va='bottom', weight='bold')

        plt.title(f"Global Balanced Accuracy")
        plt.xlabel("AL Round")
        plt.ylabel("Balanced Accuracy")
        plt.xticks(df['AL_Round'].unique())
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "9_Global_Balanced_Acc.png"), bbox_inches='tight')
        plt.close()

    # --- PLOT 10: Client Balanced Accuracy ---
    def plot_client_balanced_acc(self):
        df = self._load_client_metrics()
        if df is None or 'Balanced_Acc' not in df.columns: return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='AL_Round', y='Balanced_Acc', hue='Client_ID', 
                     palette='tab10', marker='s', linewidth=2.5)

        plt.title(f"Client Balanced Accuracy")
        plt.xlabel("AL Round")
        plt.ylabel("Balanced Accuracy")
        plt.xticks(df['AL_Round'].unique())
        plt.legend(title="Client ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "10_Client_Balanced_Acc.png"), bbox_inches='tight')
        plt.close()

    def generate_all(self, num_classes):
        print(f">>> Generating plots in {self.plots_dir}...")
        try:
            self.plot_learning_curve_detailed()
            self.plot_all_global_metrics()
            self.plot_purity_evolution()
            self.plot_global_purity_relative()
            self.plot_client_purity_relative()
            self.plot_global_balanced_acc()
            self.plot_client_balanced_acc()
            print(">>> Plotting Complete.")
        except Exception as e:
            print(f"Error during plotting: {e}")
            import traceback
            traceback.print_exc()

def auto_plot_coop_losses(base_dir, al_round):
    """
    Automatically reads CSVs for all 4 clients and saves a comparison plot.
    """
    round_dir = os.path.join(base_dir, f"AL_Round_{al_round}")
    if not os.path.exists(round_dir):
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    
    found_any = False
    for c_id in range(4):
        csv_path = os.path.join(round_dir, f"Client_{c_id}_CoOp_Loss.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sns.lineplot(data=df, x='Epoch', y='Loss', ax=axes[c_id], marker='o', color='tab:red')
            axes[c_id].set_title(f"Client {c_id} Loss")
            found_any = True
        else:
            axes[c_id].set_title(f"Client {c_id}: No Data")

    if found_any:
        plt.suptitle(f"VLM Adapter Training Loss - AL Round {al_round}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(round_dir, "coop_loss_auto_summary.png"))
    plt.close(fig) 
