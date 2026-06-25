import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
import os
import re
import json

sns.set_style("darkgrid")
mpl.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

STYLE_DICT = {
    "Random":  {"linestyle": ":",  "marker": "o", "color": "#808080"}, # Grey
    "Entropy": {"linestyle": ":", "marker": "o", "color": "#008000"}, # Green
    "FEAL":    {"linestyle": "--", "marker": "o", "color": "#F9690E"}, # Orange
    "LfOSA":   {"linestyle": "-.", "marker": "D", "color": "#8b4513"}, # SaddleBrown
    "PAL":     {"linestyle": ":", "marker": "D", "color": "#C565C7"}, # Purple
    "OpenPath": {"linestyle": "-", "marker": "o", "color": "#4B0082"}, # Dark Purple
}

def get_style(name):
    for key, attrs in STYLE_DICT.items():
        if key.lower() in name.lower(): return attrs
    
    if "Baseline" in name: return {"linestyle": "--", "marker": "X", "color": "#E6194B"} # Red
    if "Coldstart ID" in name: return {"linestyle": "--", "marker": "*", "color": "#555555"} # Dark Gray
    if "Coldstart" in name: return {"linestyle": ":", "marker": "p", "color": "#3CB44B"} # Green
    if "Global" in name: return {"linestyle": "-", "marker": "*", "color": "#4363D8"} # Blue
    if "Local" in name: return {"linestyle": "-", "marker": "*", "color": "#911EB4"} # Purple
    if "Mixed" in name: return {"linestyle": "-", "marker": "s", "color": "#56B4E9"} # Dimmer Light Blue
    
    return {"linestyle": "-", "marker": "o", "color": np.random.rand(3,)}

def plot_metric(df_agg, metric_col, title, filename, ylabel, groupby_col, envelope_type='minmax'):
    plt.figure(figsize=(5, 5))
    x = np.arange(1, 6)
    plotted = 0
    
    for name, group in df_agg.groupby(groupby_col):
        if name == "Other": continue
        
        if envelope_type == 'std':
            stats = group.groupby('Round')[metric_col].agg(['mean', 'std'])
            if len(stats) < 5 or stats['mean'].isna().all(): continue
            mult = 100 if metric_col in ["id_balanced_acc", "ood_accuracy", "ood_recall_ood", "id_purity"] else 1
            y_mean = stats['mean'].values * mult
            y_std = stats['std'].fillna(0).values * mult
            y_min = y_mean - y_std
            y_max = y_mean + y_std
        else:
            stats = group.groupby('Round')[metric_col].agg(['mean', 'min', 'max'])
            if len(stats) < 5 or stats['mean'].isna().all(): continue
            mult = 100 if metric_col in ["id_balanced_acc", "ood_accuracy", "ood_recall_ood", "id_purity"] else 1
            y_mean = stats['mean'].values * mult
            y_min = stats['min'].values * mult
            y_max = stats['max'].values * mult
        
        style = get_style(name)
        plt.fill_between(x, y_min, y_max, color=style["color"], alpha=0.2)
        plt.plot(x, y_mean, label=name, linestyle=style["linestyle"], marker=style["marker"], color=style["color"])
        plotted += 1

    if plotted > 0:
        plt.xlabel('AL Round')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x)
        
        if "Legend" in filename:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved {filename}")
    plt.close()

LOGS_DIR = 'logs_FedISIC_FarOOD_v4_fixed'
NUM_ROUNDS = 5

def parse_config_from_log(log_path):
    if not os.path.exists(log_path): return None
    try:
        with open(log_path, 'r') as f: content = f.read()
        match = re.search(r"Configuration:\s*(\{.*?\})", content, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    return None

def get_variant_type(project_name):
    if "Coldstart_ID" in project_name: return "Coldstart ID"
    if "Coldstart" in project_name: return "Coldstart"
    if "VLM_Static" in project_name: return "Baseline"
    if "G16_L0" in project_name: return "Global"
    if "G0_L16" in project_name: return "Local"
    if "G8_L8" in project_name: return "Mixed"
    return "Other"

def load_all_data():
    data = []
    for root, dirs, files in os.walk(LOGS_DIR):
        csv_files = [f for f in files if f.startswith('coop_benchmark_round') and f.endswith('.csv')]
        for csv_file in csv_files:
            round_match = re.search(r'round(\d+)\.csv', csv_file)
            if not round_match: continue
            r = int(round_match.group(1))
            
            try:
                df = pd.read_csv(os.path.join(root, csv_file))
                for _, row in df.iterrows():
                    variant = get_variant_type(row['project'])
                    if variant == "Other": continue
                    
                    data.append({
                        "Method": row['al_method'],
                        "Variant": variant,
                        "Seed": row['seed'],
                        "Round": r,
                        "Client": row['client'],
                        "id_balanced_acc": row.get('id_balanced_acc', np.nan),
                        "ood_accuracy": row.get('ood_accuracy', np.nan),
                        "ood_recall_ood": row.get('ood_recall_ood', np.nan),
                        "id_purity": row.get('id_purity', np.nan)
                    })
            except Exception as e:
                print(f"Error reading {csv_file} in {root}: {e}")
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_all_data()
    df = df[df['Round'] <= 5]
    if df.empty:
        print("No data found!")
        exit()
        
    out_dir = "plots/FedISIC"
    os.makedirs(out_dir, exist_ok=True)

    # We want to plot ID BMA, Accuracy, and OOD Recall for all active learning rounds (1-5)
    
    metrics_to_plot = {
        'id_balanced_acc': 'ID BMA (%)',
        'ood_accuracy': 'Accuracy (%)',
        'ood_recall_ood': 'OOD Recall (%)'
    }

    # 1. Overall Average Plots (filtering for "OVERALL" client)
    df_overall = df[df['Client'] == 'OVERALL']
    for metric, ylabel in metrics_to_plot.items():
        plot_metric(df_overall, metric, f'Average {ylabel}', f'{out_dir}/Avg_{metric}.png', ylabel, 'Variant')
        
        # Plot for Random only
        df_overall_random = df_overall[df_overall['Method'] == 'Random']
        plot_metric(df_overall_random, metric, f'Average {ylabel} (Random AL)', f'{out_dir}/Avg_{metric}_Random.png', ylabel, 'Variant')

    # 2. Per-Client Plots
    clients = [0, 1, 2, 3] # They are represented as '0', '1', '2', '3' or integer in CSV
    for c in clients:
        df_client = df[df['Client'] == str(c)]
        if df_client.empty:
            df_client = df[df['Client'] == float(c)]
            if df_client.empty:
                df_client = df[df['Client'] == c]
        for metric, ylabel in metrics_to_plot.items():
            plot_metric(df_client, metric, f'Client {c} {ylabel}', f'{out_dir}/Client{c}_{metric}.png', ylabel, 'Variant')
            
            # Plot for Random only
            df_client_random = df_client[df_client['Method'] == 'Random']
            plot_metric(df_client_random, metric, f'Client {c} {ylabel} (Random AL)', f'{out_dir}/Client{c}_{metric}_Random.png', ylabel, 'Variant')

    # Legend only - horizontal clean style
    from matplotlib.lines import Line2D
    variants_order = ["Baseline", "Global", "Local", "Mixed"]
    handles = []
    labels = []
    for variant in variants_order:
        style = get_style(variant)
        h = Line2D(
            [0], [0],
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"]
        )
        handles.append(h)
        labels.append(variant)

    fig = plt.figure(figsize=(10, 1))
    fig.legend(
        handles, labels,
        loc="center",
        ncol=len(handles),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/Legend_VLM_Variants.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
