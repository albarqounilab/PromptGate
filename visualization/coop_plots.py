import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
