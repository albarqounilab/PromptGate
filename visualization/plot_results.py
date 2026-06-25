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
    'axes.labelsize': 19,
    'font.size': 19,
    'legend.fontsize': 19,
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
    # AL methods
    for key, attrs in STYLE_DICT.items():
        if key.lower() in name.lower(): return attrs
    
    # VLM Variants - highly distinct colors
    if "Baseline" in name or "Static" in name: return {"linestyle": "--", "marker": "X", "color": "#E6194B"} # Red
    if "Coldstart ID" in name: return {"linestyle": "--", "marker": "*", "color": "#a6a6a6"} # Medium Gray
    if "Coldstart" in name: return {"linestyle": ":", "marker": "p", "color": "#3CB44B"} # Green
    if "Global" in name: return {"linestyle": "-", "marker": "*", "color": "#4363D8"} # Blue
    if "Local" in name: return {"linestyle": "-", "marker": "*", "color": "#911EB4"} # Purple
    if "Mixed" in name: return {"linestyle": "-", "marker": "s", "color": "#56B4E9"} # Dimmer Light Blue
    
    return {"linestyle": "-", "marker": "o", "color": np.random.rand(3,)}

def plot_envelope(df_agg, metric_col, title, filename, ylabel, groupby_col, envelope_type='minmax'):
    plt.figure(figsize=(4, 4))
    x = np.arange(1, 6)
    plotted = 0
    
    for name, group in df_agg.groupby(groupby_col):
        if name == "Other": continue
        
        if envelope_type == 'std':
            stats = group.groupby('Round')[metric_col].agg(['mean', 'std'])
            if len(stats) < 5 or stats['mean'].isna().all(): continue
            mult = 100
            y_mean = stats['mean'].values * mult
            y_std = stats['std'].fillna(0).values * mult
            y_min = y_mean - y_std
            y_max = y_mean + y_std
        elif envelope_type == 'minmax':
            stats = group.groupby('Round')[metric_col].agg(['mean', 'min', 'max'])
            if len(stats) < 5 or stats['mean'].isna().all(): continue
            mult = 100
            y_mean = stats['mean'].values * mult
            y_min = stats['min'].values * mult
            y_max = stats['max'].values * mult
        else:
            # No envelope: just calculate mean
            stats = group.groupby('Round')[metric_col].agg(['mean'])
            if len(stats) < 5 or stats['mean'].isna().all(): continue
            mult = 100
            y_mean = stats['mean'].values * mult
            y_min = y_max = None
        
        style = get_style(name)
        if envelope_type in ['std', 'minmax'] and y_min is not None:
            plt.fill_between(x, y_min, y_max, color=style["color"], alpha=0.2)
        
        plt.plot(x, y_mean, label=name, linestyle=style["linestyle"], marker=style["marker"], color=style["color"])
        plotted += 1

    if plotted > 0:
        plt.xlabel('AL Round')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x)
        
        # NO LEGEND as requested
        # plt.legend() is omitted
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
    plt.close()

LOGS_DIRS = ['logs_FedEMBED_v4_fixed', 'logs_FedEMBED_FarOOD_v4_fixed_openpath']
NUM_ROUNDS = 10
ENVELOPE_TYPE = 'none' # Set to 'minmax', 'std', or 'none'

def parse_config_from_log(log_path):
    if not os.path.exists(log_path): return None
    try:
        with open(log_path, 'r') as f: content = f.read()
        match = re.search(r"Configuration:\s*(\{.*?\})", content, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    return None

def get_variant_type(config):
    project_name = config.get('project_name', '')
    
    def get_bool(key):
        val = config.get(key, False)
        if isinstance(val, str): return val.lower() == 'true'
        return bool(val)

    is_filtered = get_bool('vlm_filter')
    g = int(config.get('coop_global_vectors', 0))
    l = int(config.get('coop_local_vectors', 0))
    
    if not is_filtered:
        if "Coldstart_ID" in project_name: return "Coldstart ID"
        return "Coldstart"

    if "Unified" in project_name:
        if g == 16 and l == 0: return "Unified G16-L0 (Global)"
        if g == 0 and l == 16: return "Unified G0-L16 (Local)"
        if g == 8 and l == 8: return "Unified G8-L8 (Mixed)"
    if "CSC" in project_name:
        if g == 16 and l == 0: return "CSC G16-L0 (Global)"
        if g == 0 and l == 16: return "CSC G0-L16 (Local)"
        if g == 8 and l == 8: return "CSC G8-L8 (Mixed)"
    if "Static" in project_name: return "VLM-Static"
    return "Other"

def load_trajectory(exp_dir, config, num_rounds):
    summary_path = os.path.join(exp_dir, "global_experiment_summary.csv")
    traj = {r: {"BMA": np.nan, "QP": np.nan, "AQR": np.nan, "Total_Labeled_Samples": np.nan} for r in range(1, num_rounds + 1)}
    
    total_id_in_dataset = None
    stats_path = os.path.join(exp_dir, "AL_Round_0", "client_data_stats.csv")
    if os.path.exists(stats_path):
        try:
            stats_df = pd.read_csv(stats_path)
            all_data_df = stats_df[stats_df['Set_Type'] == 'All_Data']
            dataset_name = config.get('dataset', 'FedISIC')
            num_id_classes = 4 if dataset_name == 'FedEMBED' else 8
            
            id_cols = [f"Class_{i}" for i in range(num_id_classes) if f"Class_{i}" in all_data_df.columns]
            if len(id_cols) > 0:
                total_id_in_dataset = all_data_df[id_cols].sum().sum()
        except: pass

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            if not df.empty and 'AL_Round' in df.columns:
                for _, row in df.iterrows():
                    r = int(row['AL_Round'])
                    if r >= 1 and r <= num_rounds:
                        if 'Avg_Bal_Acc' in row: traj[r]["BMA"] = row['Avg_Bal_Acc']
                        if 'Avg_QR' in row: traj[r]["QP"] = row['Avg_QR']
                        elif 'Avg_QP' in row: traj[r]["QP"] = row['Avg_QP']
                        if 'Total_Labeled_Samples' in row:
                            traj[r]["Total_Labeled_Samples"] = row['Total_Labeled_Samples']
                        if 'Avg_AQR' in row: 
                            traj[r]["AQR"] = row['Avg_AQR']
        except: pass
        
    if total_id_in_dataset is not None and total_id_in_dataset > 0:
        for r in range(1, num_rounds + 1):
            if not pd.isna(traj[r]["Total_Labeled_Samples"]):
                traj[r]["AQR"] = traj[r]["Total_Labeled_Samples"] / total_id_in_dataset

    return traj

def load_all_data(configs):
    data = []
    for cfg in configs:
        l_dirs = cfg['dirs']
        num_rounds = cfg['rounds']
        dataset_name = cfg['name']
        for l_dir in l_dirs:
            if not os.path.exists(l_dir): continue
            for root, dirs, files in os.walk(l_dir):
                if "experiment_log.txt" in files:
                    config = parse_config_from_log(os.path.join(root, "experiment_log.txt"))
                    if not config: continue
                    variant = get_variant_type(config)
                    if variant == "Other": continue
                    
                    method = config.get('al_method', 'Unknown')
                    seed = int(config.get('seed', 0))
                    traj = load_trajectory(root, config, num_rounds)
                    
                    for r in range(1, num_rounds + 1):
                        data.append({
                            "Dataset": dataset_name, "Method": method, "Variant": variant, "Seed": seed, "Round": r,
                            "BMA": traj[r]["BMA"], "QP": traj[r]["QP"], "AQR": traj[r]["AQR"], "MaxRound": num_rounds
                        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    CONFIGS = [
        {
            "name": "FedISIC",
            "dirs": ['/albarqouni_lab/lab_users/dgaviria/repositories/test/vlm_fidal/logs_FedISIC_FarOOD_v4_fixed'],
            "rounds": 5,
            "out": "./plots/FedISICFarOOD"
        },
        {
            "name": "FedEMBED",
            "dirs": ['/albarqouni_lab/lab_users/dgaviria/repositories/test/vlm_fidal/logs_FedEMBED_v4_fixed'],
            "rounds": 10,
            "out": "./plots/FedEMBED_fixed"
        }
    ]

    full_df = load_all_data(CONFIGS)
    
    #allowed_methods = ['Random', 'Entropy', 'FEAL', 'PAL', 'LfOSA', 'OpenPath', 'EOAL', 'Badge']
    allowed_methods = ['Random', 'Entropy', 'FEAL', 'PAL', 'LfOSA', 'OpenPath']
    full_df = full_df[full_df['Method'].isin(allowed_methods)]
    
    # Also filter out unknown/unwanted variants
    allowed_variants = ['VLM-Static', 'Coldstart', 'Coldstart ID', 'CSC G16-L0 (Global)', 'CSC G0-L16 (Local)', 'CSC G8-L8 (Mixed)']
    full_df = full_df[full_df['Variant'].isin(allowed_variants)]
    
    ENVELOPE_TYPE = 'none'

    for cfg in CONFIGS:
        df_full = full_df[full_df['Dataset'] == cfg['name']]
        if df_full.empty: continue
        
        # We'll generate both shaded and non-shaded versions
        for shading in ['none', 'minmax']:
            suffix = "shaded" if shading == 'minmax' else "noshading"
            out_root = f"plots/{cfg['name']}_{suffix}"
            os.makedirs(out_root, exist_ok=True)
            num_rounds = cfg['rounds']
            
            # Localized plotting function to handle shading
            def plot_dataset_local(df_agg, metric_col, title, filename, ylabel, groupby_col, nr, envelope=shading):
                plt.figure(figsize=(4, 4))
                x = np.arange(1, nr + 1)
                plotted = 0
                for name, group in df_agg.groupby(groupby_col):
                    if name == "Other": continue
                    
                    if envelope == 'minmax':
                        stats = group.groupby('Round')[metric_col].agg(['mean', 'min', 'max'])
                        y_min = stats['min'].values * 100
                        y_max = stats['max'].values * 100
                        if len(y_min) > nr: y_min = y_min[:nr]
                        if len(y_max) > nr: y_max = y_max[:nr]
                    else:
                        stats = group.groupby('Round')[metric_col].agg(['mean'])
                    
                    if stats['mean'].isna().all(): continue
                    y_mean = stats['mean'].values * 100
                    if len(y_mean) > nr: y_mean = y_mean[:nr]
                    if len(y_mean) < nr: continue
                    
                    style = get_style(name)
                    z = 1 if ("Baseline" in name or "Static" in name or "Random" in name) else 5
                    
                    if envelope == 'minmax' and len(y_mean) == len(y_min):
                        plt.fill_between(x, y_min, y_max, color=style["color"], alpha=0.2, zorder=z)
                    
                    plt.plot(x, y_mean, label=name, linestyle=style["linestyle"], marker=style["marker"], color=style["color"], zorder=z)
                    plotted += 1
                
                if plotted > 0:
                    # Annotation logic: Improvement of (Global/Local/Mixed) over second best (excluding Coldstart ID)
                    if groupby_col == 'Legend_Variant':
                        variants_of_interest = ["Global", "Local", "Mixed"]
                        other_variants = ["Static VLM", "Coldstart"] # Exclude Coldstart ID
                        all_variants_to_compare = variants_of_interest + other_variants
                        
                        round_data = {}
                        # Re-calculate means per round for comparison
                        for r in x:
                            round_data[r] = {}
                            for name, group in df_agg.groupby(groupby_col):
                                if name in all_variants_to_compare:
                                    stats = group[group['Round'] == r][metric_col]
                    plt.xlabel('AL Round')
                    plt.ylabel(ylabel)
                    plt.title(title)
                    plt.xticks(x)
                    plt.tight_layout()
                    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
                    print(f"Saved {filename}")
                plt.close()

            # 1. Compare VLM Variants
            mapping = {
                "VLM-Static": "Static VLM",
                "Coldstart ID": "Coldstart ID",
                "Coldstart": "Coldstart",
                "CSC G16-L0 (Global)": "Global",
                "CSC G0-L16 (Local)": "Local",
                "CSC G8-L8 (Mixed)": "Mixed",
            }
            df_full['Legend_Variant'] = df_full['Variant'].map(mapping)
            df_vlm = df_full.dropna(subset=['Legend_Variant'])
            plot_dataset_local(df_vlm, 'BMA', '', f'{out_root}/BMA_VLM_Variants.png', 'BMA (%)', 'Legend_Variant', num_rounds)
            plot_dataset_local(df_vlm, 'QP', '', f'{out_root}/QP_VLM_Variants.png', 'QP (%)', 'Legend_Variant', num_rounds)
            plot_dataset_local(df_vlm, 'AQR', '', f'{out_root}/AQR_VLM_Variants.png', 'AQR (%)', 'Legend_Variant', num_rounds)

            # 2. Compare AL Methods for Mixed
            best_variant = "CSC G8-L8 (Mixed)"
            df_al = df_full[df_full['Variant'] == best_variant]
            v_name = best_variant.replace(" ", "_").replace("(", "").replace(")", "")
            plot_dataset_local(df_al, 'BMA', '', f'{out_root}/BMA_AL_{v_name}.png', 'BMA (%)', 'Method', num_rounds)
            plot_dataset_local(df_al, 'QP', '', f'{out_root}/QP_AL_{v_name}.png', 'QP (%)', 'Method', num_rounds)
            plot_dataset_local(df_al, 'AQR', '', f'{out_root}/AQR_AL_{v_name}.png', 'AQR (%)', 'Method', num_rounds)

            # 3. Compare AL Methods for Baseline
            df_base = df_full[df_full['Legend_Variant'] == "Static VLM"]
            plot_dataset_local(df_base, 'BMA', '', f'{out_root}/BMA_AL_Baseline.png', 'BMA (%)', 'Method', num_rounds)
            plot_dataset_local(df_base, 'QP', '', f'{out_root}/QP_AL_Baseline.png', 'QP (%)', 'Method', num_rounds)
            plot_dataset_local(df_base, 'AQR', '', f'{out_root}/AQR_AL_Baseline.png', 'AQR (%)', 'Method', num_rounds)

            # Generate horizontal VLM Variant Legend
            from matplotlib.lines import Line2D
            v_order = ["Global", "Local", "Mixed", "Baseline", "Coldstart", "Coldstart ID"]
            handles = []
            labels = []
            for vn in v_order:
                # Map back to get style if needed, or if style dict handles it
                # get_style works for "Baseline", so need to align
                s_name = "Baseline" if vn == "Static VLM" else vn
                style = get_style(s_name)
                handles.append(Line2D([0], [0], color=style["color"], lw=2, linestyle=style["linestyle"], marker=style["marker"]))
                labels.append(vn)
            
            fig_leg = plt.figure(figsize=(14.5, 0.5))
            ax_leg = fig_leg.add_subplot(111)
            ax_leg.axis('off')
            ax_leg.legend(handles, labels, loc='center', ncol=6, frameon=False, fontsize=18)
            plt.savefig(f'{out_root}/Legend_VLM_Variants.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

            # Generate horizontal AL Method Legend
            al_order = ["Random", "Entropy", "FEAL", "LfOSA", "PAL", "OpenPath"]
            al_handles = []
            al_labels = []
            for an in al_order:
                style = get_style(an)
                al_handles.append(Line2D([0], [0], color=style["color"], lw=2, linestyle=style["linestyle"], marker=style["marker"]))
                al_labels.append(an)
            
            fig_al_leg = plt.figure(figsize=(12, 0.5))
            ax_al_leg = fig_al_leg.add_subplot(111)
            ax_al_leg.axis('off')
            ax_al_leg.legend(al_handles, al_labels, loc='center', ncol=len(al_order), frameon=False, fontsize=18)
            plt.savefig(f'{out_root}/Legend_AL_Methods.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

