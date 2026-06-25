import os
import glob
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

sns.set_style("darkgrid")
mpl.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 19,
    'font.size': 19,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 9
})

STYLE_DICT = {
    4:   {"linestyle": ":",  "marker": "v", "color": "#008000"}, # Green
    8:   {"linestyle": ":",  "marker": "^", "color": "#56B4E9"}, # Light Blue
    16:  {"linestyle": "--", "marker": "o", "color": "#F7CA18"}, # Yellow
    32:  {"linestyle": "--", "marker": "s", "color": "#F9690E"}, # Orange
    128: {"linestyle": "-",  "marker": "D", "color": "#E6194B"}, # Red
}

ALLOWED_METHODS = ['Random', 'Entropy', 'FEAL', 'PAL', 'LfOSA', 'OpenPath']

def parse_coop_shots(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    match = re.search(r'"coop_shots":\s*"(\d+)"', content)
    if match:
        return int(match.group(1))
    return None

def get_total_id_from_csv(csv_path):
    """Reads the hardcoded CSV to find the total ID samples for FedISIC (8 classes)."""
    if not os.path.exists(csv_path):
        print(f"Warning: Hardcoded CSV not found at {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        all_data = df[df['Set_Type'] == 'All_Data']
        # FedISIC has 8 ID classes
        id_cols = [f"Class_{i}" for i in range(8) if f"Class_{i}" in all_data.columns]
        if not id_cols:
            return None
        total_id = all_data[id_cols].sum().sum()
        return total_id
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def main():
    # Hardcoded path to extract the denominator
    HARDCODED_STATS_CSV = "logs_FedISIC_15_epochs_shots/FedISIC/Random/seed_0/2026-02-20_03-14-50/AL_Round_0/client_data_stats.csv"
    total_id_samples = get_total_id_from_csv(HARDCODED_STATS_CSV)
    
    if total_id_samples:
        print(f"Extracted Total ID Samples for AQR calculation: {total_id_samples}")
    else:
        print("Could not extract Total ID Samples. AQR recalculation may fail.")

    base_dir = "logs_FedISIC_15_epochs_shots/FedISIC"
    log_files = glob.glob(os.path.join(base_dir, "*", "seed_*", "*", "experiment_log.txt"))
    
    all_data = []
    
    for log_file in log_files:
        run_dir = os.path.dirname(log_file)
        summary_file = os.path.join(run_dir, "global_experiment_summary.csv")
        
        if not os.path.exists(summary_file):
            continue
            
        coop_shots = parse_coop_shots(log_file)
        if coop_shots is None:
            continue
            
        df = pd.read_csv(summary_file)
        if df.empty:
            continue
            
        df['coop_shots'] = coop_shots
        parts = run_dir.split(os.sep)
        try:
            method_idx = parts.index("FedISIC") + 1
            method = parts[method_idx]
            seed_str = parts[method_idx + 1]
            seed = int(seed_str.split('_')[1])
        except:
            method = "Unknown"
            seed = 0
            
        # Keep only the methods that match the allowed list (case-insensitive)
        is_allowed = any(allowed.lower() in method.lower() for allowed in ALLOWED_METHODS)
        if not is_allowed:
            continue
            
        df['method'] = method
        df['seed'] = seed
        
        all_data.append(df)
        
    if not all_data:
        print("No data found!")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    out_dir = "plots/ablation_shots"
    os.makedirs(out_dir, exist_ok=True)
    
    metrics_to_plot = {
        'Avg_Bal_Acc': 'BMA (%)',
        'ID_Purity': 'QP (%)',
        'Avg_AQR': 'AQR (%)'
    }
    
    # Create Legend Separately
    fig_leg, ax_leg = plt.subplots(figsize=(1, 1))
    ax_leg.axis('off')
    handles = []
    labels = []
    from matplotlib.lines import Line2D
    for shots in sorted(full_df['coop_shots'].unique()):
        style = STYLE_DICT.get(shots, {"linestyle": "-", "marker": "o", "color": "black"})
        line = Line2D([0], [0], label=f"{shots} Shots", color=style['color'], 
                      linestyle=style['linestyle'], marker=style['marker'], 
                      markersize=9, linewidth=2.5)
        handles.append(line)
        labels.append(f"{shots} Shots")
    
    legend = ax_leg.legend(handles, labels, loc='center', ncol=len(labels), frameon=False, fontsize=18)
    
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    
    fig_leg.savefig(os.path.join(out_dir, "legend_shots.png"), dpi=300, bbox_inches=bbox.expanded(1.1, 1.2), transparent=True)
    plt.close(fig_leg)

    # Convert to percentages for the plots
    plot_df = full_df.copy()
    if 'Avg_Bal_Acc' in plot_df.columns: plot_df['Avg_Bal_Acc'] *= 100
    if 'Avg_AUC' in plot_df.columns: plot_df['Avg_AUC'] *= 100
    if 'ID_Purity' in plot_df.columns: plot_df['ID_Purity'] *= 100
    
    # --- Recalculate AQR using the hardcoded denominator ---
    if total_id_samples is not None and 'Total_Labeled_Samples' in plot_df.columns:
        plot_df['Avg_AQR'] = (plot_df['Total_Labeled_Samples'] / total_id_samples) * 100
    else:
        # Fallback if Total_Labeled_Samples is missing from the new logs
        print("Warning: 'Total_Labeled_Samples' column missing. Using existing Avg_AQR without recalculation.")

    # Average metrics over methods first, so we use Seed to calculate std/min/max
    df_averaged_methods = plot_df.groupby(['coop_shots', 'AL_Round', 'seed']).mean(numeric_only=True).reset_index()

    # Plot each metric
    for metric, title in metrics_to_plot.items():
        if metric not in df_averaged_methods.columns:
            continue
            
        plt.figure(figsize=(4.4, 4.4))
        
        for shots, group in df_averaged_methods.groupby('coop_shots'):
            stats = group.groupby('AL_Round')[metric].agg(['mean', 'min', 'max'])
            
            rounds_present = stats.index.values
            y_mean = stats['mean'].values
            y_min = stats['min'].values
            y_max = stats['max'].values
            
            style = STYLE_DICT.get(shots, {"linestyle": "-", "marker": "o", "color": "black"})
            
            plt.fill_between(rounds_present, y_min, y_max, color=style["color"], alpha=0.2)
            plt.plot(rounds_present, y_mean, linestyle=style["linestyle"], marker=style["marker"], color=style["color"])
            
        plt.xlabel('AL Round')
        plt.ylabel(title)
        
        plt.xticks(np.arange(1, 6))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_over_rounds.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    # Generate Table (Using decimals instead of %)
    max_round = full_df['AL_Round'].max()
    final_df = plot_df[plot_df['AL_Round'] == max_round] # Ensure table uses the recalculated values
    
    # We want mean across methods AND seeds for the final table.
    table_df = final_df.groupby('coop_shots')[list(metrics_to_plot.keys())].agg(['mean', 'std'])
    
    formatted_table = pd.DataFrame()
    for metric in metrics_to_plot.keys():
        if metric in table_df.columns.levels[0]:
            formatted_table[metric] = table_df[metric]['mean'].map('{:.1f}'.format)
            
    formatted_table.reset_index(inplace=True)
    formatted_table.rename(columns={'coop_shots': 'Shots'}, inplace=True)
    
    with open(os.path.join(out_dir, "ablation_table_final_round.md"), "w") as f:
        f.write(f"### Ablation Study: Effect of CoOp Shots (Final AL Round: {max_round})\n\n")
        f.write("*Note: Metrics are averaged across all AL methods and seeds.*\n\n")
        headers = list(formatted_table.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for _, row in formatted_table.iterrows():
            f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")

if __name__ == "__main__":
    main()