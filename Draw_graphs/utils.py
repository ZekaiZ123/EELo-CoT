import torch
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random

import torch
import pandas as pd
import re
import regex
import multiprocessing

from datasets import load_dataset
from sympy.parsing.latex import parse_latex
from sympy import simplify, N


from math import isclose
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex

import json
import sympy as sp
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from collections import defaultdict





def figure_layer():
    plt.style.use('ggplot')               
    plt.rcParams.update({
        'font.size': 12,                         
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
    })

    # 2. 数据读取
    model_info = {
        "Qwen2.5-7b":             "/data/LLM_neuron_data/argmax/math500/Qwen2.5_7b/layer",
        "Qwen2.5-7b-Instruct":    "/data/LLM_neuron_data/argmax/math500/Qwen2.5_7b_Instruct/layer",
        "Qwen2.5-Math-7b":        "/data/LLM_neuron_data/argmax/math500/Qwen2.5_Math_7b/layer10000",
        "Qwen2.5-Math-7b-Instruct": "/data/LLM_neuron_data/argmax/math500/Qwen2.5_Math_7b_Instruct/layer",
    }

    counts = {}
    for name, base_dir in model_info.items():
        layer_counts = []
        for layer_idx in range(28):
            csv_path = os.path.join(base_dir, f"from670_on_math500_len{layer_idx}.csv")
            df = pd.read_csv(csv_path)
            layer_counts.append((df["act_val"] > 1).sum())
        counts[name] = layer_counts

    # 3. draw figure
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')

    # custom color
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (name, layer_counts), c in zip(counts.items(), colors):
        ax.plot(
            range(28), 
            layer_counts, 
            marker='o', 
            linestyle='-', 
            linewidth=2, 
            markersize=6, 
            color=c, 
            label=name
        )

    # 4. set title and label
    ax.set_title("各模型 Layer 上 act>1 的神经元数量对比")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Count of activations > 1")

    # hide top and right spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # show y axis grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(False)

    # 5. set legend
    leg = ax.legend(frameon=True)
    leg.get_frame().set_alpha(0.8)

    # 6. save figure
    plt.tight_layout()
    save_path = "number_>1_comparison.png"
    plt.savefig(save_path, dpi=300, facecolor='white')

    plt.show()



def plot_neuron_trends(
    pt_path: str,
    prompt_idx: int,
    neuron_list: list[int],
    output_path: list[int],
    show_tokens: bool = True
):

    # 1. load data
    data = torch.load(pt_path)
    all_neurons = data["neuron_ids"]
    results    = data["results"]

    # 2. find the record of the corresponding prompt
    rec = next((r for r in results if r["Index"] == prompt_idx), None)
    if rec is None:
        raise ValueError(f"没有找到 Index = {prompt_idx} 的 prompt 记录。")
    tokens      = rec["tokens"]        # List[str], length = seq_len
    activations = rec["activations"]   # Tensor(seq_len, n_selected)

    # 3. calculate the column index to plot
    #    the index of each neuron in neuron_list in all_neurons
    col_idxs = []
    for nid in neuron_list:
        if nid not in all_neurons:
            raise ValueError(f"Neuron {nid} 不在数据的 neuron_ids 列表中。")
        col_idxs.append(all_neurons.index(nid))

    seq_len, _ = activations.shape

    # 4. draw figure
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("white")

    # cycle color
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (nid, col) in enumerate(zip(neuron_list, col_idxs)):
        act = activations[:, col].cpu().numpy()  # (seq_len,)
        ax.plot(
            range(seq_len),
            act,
            # marker="o",
            linestyle="-",
            linewidth=1,
            markersize=5,
            color=color_cycle[i % len(color_cycle)],
            label=f"Neuron {nid} ({i*10})"
        )

    ax.set_xlabel("Token Position (step)", fontsize=12)
    ax.set_ylabel("Activation Value", fontsize=12)
    ax.set_title(f"Prompt Index {prompt_idx} — len Neuron Activation Trends", fontsize=14)

    # hide top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # show y axis grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.grid(False)


    if show_tokens:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_xticks(range(seq_len))

    ax.legend(frameon=True, fontsize=10).get_frame().set_alpha(0.8)
    plt.tight_layout()
    # save_path = "step4.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"figure saved to {output_path}")
    plt.show()



def extract_high_activation_tokens(
    pt_path: str,
    # neuron_ids: list[int],
    output_csv: str,
    window_size: int = 10,
    threshold = 4
):

    # 1. load data
    data = torch.load(pt_path)
    all_neurons = data["neuron_ids"]
    results = data["results"]  # List of dicts with keys: "Index", "tokens", "activations"

    rows = []
    for rec in results:
        prompt_idx = rec.get("Index", None)
        tokens = rec["tokens"]              # List[str], length = seq_len
        activations = rec["activations"]    # Tensor of shape (seq_len, n_selected)

        seq_len, n_selected = activations.shape
        # for each selected neuron
        for j, nid in enumerate(all_neurons):
            # if nid not in neuron_ids:
            #     continue
            act_seq = activations[:, j].numpy()  # shape (seq_len,)

            for idx in (act_seq > threshold).nonzero()[0]:
                start = max(0, idx - window_size)
                end = min(idx + window_size, len(tokens))
                token_window = tokens[start:end]
                # window_str = " ".join(token_window)
                window_str = " ".join(tok.replace("Ġ", "") for tok in token_window)
                rows.append({
                    "Neuron_ID": nid,
                    "Prompt_Index": prompt_idx,
                    "Token_Position": idx,
                    "Token_Window": window_str,
                    "Activation_Value": float(act_seq[idx]),
                    "target_token": tokens[idx].replace("Ġ", "")
                })

  
    df_out = pd.DataFrame(rows, columns=[
        "Neuron_ID", "Prompt_Index", "Token_Position", "Token_Window", "Activation_Value", "target_token"
    ])

    # os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)


def compute_neuron_token_averages(
    pt_path: str,
    output_csv: str
):

    # 1. load data
    data = torch.load(pt_path)
    neuron_ids = data["neuron_ids"]
    results    = data["results"]

    agg = { nid: defaultdict(list) for nid in neuron_ids }

    for rec in tqdm(results, desc="Processing prompts", unit="prompt"):
        tokens   = rec["tokens"]
        acts     = rec["activations"]  # Tensor(seq_len, n_selected)
        seq_len, _ = acts.shape


        for pos, tok in enumerate(tokens):
            vec = acts[pos]  # Tensor(n_selected,)
            for k, nid in enumerate(neuron_ids):
                agg[nid][tok].append(vec[k].item())

    rows = []
    for nid in tqdm(neuron_ids, desc="Aggregating neurons", unit="neuron"):
        for tok, vals in agg[nid].items():
            cnt = len(vals)
            avg = sum(vals) / cnt
            rows.append({
                "Neuron_ID":      nid,
                "Token":          tok,
                "Avg_Activation": avg,
                "Count":          cnt
            })

    # 5. 保存到 CSV
    df = pd.DataFrame(rows)
    df = df.sort_values(["Neuron_ID","Avg_Activation"], ascending=[True, False])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"{output_csv} {len(df)} rows saved")



def top_wait_neurons_from_csv(
    input_csv: str,
    output_csv: str,
    top_k: int = 50
):

    df = pd.read_csv(input_csv)

    mask = df["Token"] == "Wait"
    df_wait = df[mask].copy()


    df_top = df_wait.sort_values("Avg_Activation", ascending=False).head(top_k)


    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_top.to_csv(output_csv, index=False)

    return df_top

def generate_activation_heatmap(input_file_path, output_image_path):

    df = pd.read_csv(input_file_path)

    activation_counts = df.groupby(['Neuron_ID', 'Prompt_Index']).size().reset_index(name='Activation_Count')

    activation_counts['Neuron_ID'] = pd.to_numeric(activation_counts['Neuron_ID'], errors='coerce')  
    activation_counts = activation_counts.sort_values(by='Neuron_ID', ascending=False)  

    heatmap_data = activation_counts.pivot_table(index='Neuron_ID', columns='Prompt_Index', values='Activation_Count', fill_value=0)

    heatmap_data_log = np.log1p(heatmap_data) 


    plt.figure(figsize=(12, 8))  
    sns.heatmap(heatmap_data_log, annot=False, cmap='plasma') 
    plt.title('Neuron Activation Count Heatmap (Log Transformed)')  
    plt.xlabel('Prompt Index')  
    plt.ylabel('Neuron ID')  
    plt.savefig(output_image_path, dpi=300)  
    plt.close()  


def count_tokens_and_per_neuron(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path, dtype={'target_token': str})
    grp = (
        df
        .groupby(['target_token', 'Neuron_ID'])
        .size()
        .reset_index(name='neuron_count')
    )
    wide = (
        grp
        .pivot(index='target_token', columns='Neuron_ID', values='neuron_count')
        .fillna(0)            
        .astype(int)          
    )

    wide['total_count'] = wide.sum(axis=1)

    neuron_cols = sorted(c for c in wide.columns if c != 'total_count')
    wide = wide[['total_count'] + neuron_cols]

    wide = wide.sort_values('total_count', ascending=False)

    wide = wide.reset_index().rename(columns={'target_token': 'token'})

    wide.to_csv(output_csv_path, index=False, encoding='utf-8')



def plot_digit_context_activation(
    pt_path: str,
    neuron_csv: str,
    window_before: int = 50,
    window_after: int = 100,
    output_path: str = "digit_context_activation.png"
):
    data = torch.load(pt_path)
    neuron_ids = data["neuron_ids"]
    results = data["results"]

    sel_df = pd.read_csv(neuron_csv)
    selected_neurons = sel_df['Neuron_ID'].astype(int).tolist()

    col_idxs = [neuron_ids.index(n) for n in selected_neurons]

    offsets = list(range(-window_before, window_after + 1))
    digits = [str(d) for d in range(10)]
    digits.append("Wait")

    digit_windows = {d: [] for d in digits}
    for rec in tqdm(results, desc="Scanning prompts"):
        tokens = rec["tokens"]
        acts = rec["activations"][:, col_idxs].cpu().numpy()  
        seq_len = acts.shape[0]
        for pos, tok in enumerate(tokens):
            if tok in digit_windows:
                window_vals = []
                for off in offsets:
                    idx = pos + off
                    if 0 <= idx < seq_len:
                        window_vals.append(np.mean(acts[idx]))
                    else:
                        window_vals.append(np.nan)
                digit_windows[tok].append(window_vals)

    avg_activations = {}
    for d in digits:
        arr = np.array(digit_windows[d], dtype=float)  
        avg_activations[d] = np.nanmean(arr, axis=0) 


    plt.figure(figsize=(10, 6), facecolor='white')
    for d in digits:
        plt.plot(offsets, avg_activations[d], marker='o', markersize=3, label=f"'{d}'")
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Position relative to digit token")
    plt.ylabel("Avg activation (across selected neurons)")
    plt.title("Contextual activation around digit tokens (0–9)")
    plt.legend(title="Digit")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

from scipy.optimize import curve_fit

def plot_wait_decay_with_log_fit(
    pt_path: str,
    neuron_csv: str,
    window_before: int = 50,
    window_after: int = 100,
    output_path: str = "wait_context_decay.png"
):
    data = torch.load(pt_path)
    neuron_ids = data["neuron_ids"]
    results = data["results"]

    sel_df = pd.read_csv(neuron_csv)
    selected_neurons = sel_df['Neuron_ID'].astype(int).tolist()
    col_idxs = [neuron_ids.index(n) for n in selected_neurons]

    offsets = np.arange(-window_before, window_after+1)
    digit = "Wait"
    windows = []

    for rec in tqdm(results, desc="Scanning prompts for 'Wait'"):
        tokens = rec["tokens"]
        acts   = rec["activations"][:, col_idxs].cpu().numpy()
        L = acts.shape[0]
        for pos, tok in enumerate(tokens):
            if tok == digit:
                vals = []
                for off in offsets:
                    idx = pos + off
                    if 0 <= idx < L:
                        vals.append(np.mean(acts[idx]))
                    else:
                        vals.append(np.nan)
                windows.append(vals)


    arr = np.array(windows, dtype=float)        
    mean_curve = np.nanmean(arr, axis=0)         


    xs = offsets[offsets > 0]
    ys = mean_curve[offsets > 0]

    def decay_func(x, a, b, c):
        return a - b * np.log(x + c)


    p0 = [ys[0], 1.0, 1.0]
    popt, _ = curve_fit(decay_func, xs, ys, p0=p0, maxfev=10000)


    plt.figure(figsize=(10, 6), facecolor='white')
    plt.plot(offsets, mean_curve, marker='o', markersize=4, label="Empirical 'Wait'")
    fit_curve = decay_func(xs, *popt)
    plt.plot(xs, fit_curve, color='red', linewidth=2,
             label=f"Fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}")

    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Position relative to 'Wait'")
    plt.ylabel("Avg activation (log scale)")
    plt.title("Decay of 'Wait' activation and −log fit")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")
    plt.show()




def fit_and_plot_proportional_abc(
    pt_path: str,
    neuron_csv: str,
    window_before: int = 50,
    window_after: int = 100,
    output_path: str = "wait_decay_prop_abc.png"
):

    data = torch.load(pt_path)
    neuron_ids = data["neuron_ids"]
    results    = data["results"]
    sel_df     = pd.read_csv(neuron_csv)
    selected   = sel_df["Neuron_ID"].astype(int).tolist()
    col_idxs   = [neuron_ids.index(n) for n in selected]


    offsets = np.arange(-window_before, window_after+1)
    windows = []
    for rec in tqdm(results, desc="Collecting 'Wait' windows"):
        tokens = rec["tokens"]
        acts   = rec["activations"][:, col_idxs].cpu().numpy()
        L      = acts.shape[0]
        for pos, tok in enumerate(tokens):
            if tok == "Wait":
                vals = [ np.mean(acts[pos+off]) if 0<=pos+off<L else np.nan
                         for off in offsets ]
                windows.append(vals)


    arr = np.array(windows, float)
    mean_curve = np.nanmean(arr, axis=0)
    M0 = mean_curve[offsets==0][0]


    xs = offsets[offsets>=1]
    ys = mean_curve[offsets>=1]
    def model(x, a, b, c):
        return a - b * np.log(x + c)
    popt, _ = curve_fit(model, xs, ys, p0=[ys[0], 1.0, 1.0], maxfev=10000)
    a_fit, b_fit, c_fit = popt

    m = a_fit / M0
    n = b_fit / M0
    l = c_fit / M0


    print("M0 (mean_curve[0]) =", M0)
    print(f"Fitted a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}")
    print(f"Therefore: m = a/M0 = {m:.4f}, n = b/M0 = {n:.4f}, l = c/M0 = {l:.4f}")
    print(f"Check: m*M0 = {m*M0:.4f}, n*M0 = {n*M0:.4f}, l*M0 = {l*M0:.4f}")


    plt.figure(figsize=(10,6), facecolor='white')
    plt.plot(offsets, mean_curve, 'o-', ms=4, label="Empirical 'Wait'")
    fit_vals = model(xs, *popt)
    plt.plot(xs, fit_vals, '-', lw=2, color='crimson',
             label=f"Fit: a={m:.3f}·M0, b={n:.3f}·M0, c={l:.3f}·M0")
    plt.axvline(0, color='gray', ls='--')
    plt.xlabel("Position relative to 'Wait'")
    plt.ylabel("Avg activation (log scale)")
    plt.title("Decay of 'Wait' Activation (a,b,c ∝ M0)")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()


def process_pt_file(
    pt_file_path: str,
    csv_output_path: str,
    threshold: float = 4.0,
    min_occurrences: int = 1
) -> "pandas.DataFrame":
    df = pd.read_csv(input_csv)

    ids = df.loc[df["Category"] == category, ex_type]
    with open(output_txt, "w") as f:
        for nid in ids:
            f.write(f"{nid}\n")


import glob

def plot_acc_neuron_counts(
    folder_path: str,
    threshold,
    file_prefix: str = "from670_on_math500_acc",
    file_suffix: str = "_2000.csv",
    output_path: str = "acc_active_counts.png"
):

    pattern = os.path.join(folder_path, f"{file_prefix}*{file_suffix}")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No files match {pattern}")

    layer_idxs = []
    counts = []
    for fn in sorted(files, key=lambda f: int(re.search(
            rf"{re.escape(file_prefix)}(\d+){re.escape(file_suffix)}", f
        ).group(1))):
        m = re.search(rf"{re.escape(file_prefix)}(\d+){re.escape(file_suffix)}", fn)
        layer = int(m.group(1))
        df = pd.read_csv(fn)
        cnt = (df["act_val"] > threshold).sum()
        layer_idxs.append(layer)
        counts.append(cnt)


    plt.figure(figsize=(8, 5))
    plt.plot(layer_idxs, counts, marker="o", linestyle="-")
    plt.xlabel("Layer index")
    plt.ylabel(f"Neurons with act_val > {threshold}")
    plt.title("Active neurons per layer")
    plt.xticks(layer_idxs) 
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()



def extract_global_top_neurons(
    folder_path: str,
    file_prefix: str = "from670_on_math500_acc",
    file_suffix: str = "_2000.csv",
    top_n: int = 30,
    output_csv: str = "top30_neurons_across_layers.csv"
):
    pattern = os.path.join(folder_path, f"{file_prefix}*{file_suffix}")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match pattern {pattern}")
    dfs = []
    for fn in files:
        basename = os.path.basename(fn)
        m = re.match(rf"{re.escape(file_prefix)}(\d+){re.escape(file_suffix)}", basename)
        if not m:
            continue
        layer = int(m.group(1))
        df = pd.read_csv(fn)
        df["layer"] = layer
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    top_df = all_df.nlargest(top_n, "act_val")
    top_df = top_df[["Neuron_ID", "act_val", "layer", "Category"]]
    top_df.to_csv(output_csv, index=False)




def plot_acc_neuron_counts_3model(
    folder_path1: str,
    folder_path2: str,
    folder_path3: str,
    threshold: float,
    output_path: str = "acc_active_counts.png"
):

    folder_paths = [folder_path1, folder_path2]
    counts_list = []
    layer_idxs = None

    for fp in folder_paths:
        pre = "from670_on_math500_acc"
        suffix  = "_2000.csv"
        pattern = os.path.join(fp, f"from670_on_math500_acc*_2000.csv")
        files = glob.glob(pattern)
        if not files:
            raise ValueError(f"No files match {pattern}")
        sorted_files = sorted(
            files,
            key=lambda f: int(re.search(
                rf"{re.escape(pre)}(\d+){re.escape(suffix)}", f
            ).group(1))
        )
        layers = []
        counts = []
        for fn in sorted_files:
            m = re.search(rf"{re.escape(pre)}(\d+){re.escape(suffix)}", fn)
            layer = int(m.group(1))
            df = pd.read_csv(fn)
            cnt = (df["act_val"] > threshold).sum()
            layers.append(layer)
            counts.append(cnt)
        if layer_idxs is None:
            layer_idxs = layers
        counts_list.append(counts)

    pre = "math500_acc"
    suffix  = "_2000.csv"

    pattern = os.path.join(folder_path3, f"math500_acc*_2000.csv")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No files match {pattern}")
    sorted_files = sorted(
        files,
        key=lambda f: int(re.search(
            rf"{re.escape(pre)}(\d+){re.escape(suffix)}", f
        ).group(1))
    )
    layers = []
    counts = []
    for fn in sorted_files:
        m = re.search(rf"{re.escape(pre)}(\d+){re.escape(suffix)}", fn)
        layer = int(m.group(1))
        df = pd.read_csv(fn)
        cnt = (df["act_val"] > threshold).sum()
        layers.append(layer)
        counts.append(cnt)
    if layer_idxs is None:
        layer_idxs = layers
    counts_list.append(counts)

    plt.rcParams['axes.titlesize']     = 25
    plt.rcParams['axes.labelsize']     = 25
    plt.rcParams['xtick.labelsize']    = 25
    plt.rcParams['ytick.labelsize']    = 25
    plt.rcParams['legend.fontsize']    = 25
    # 绘制三条折线
    plt.figure(figsize=(8, 6))
    labels = ["R1-distill-Qwen-7B", "Qwen2.5-Math-7B-base", "Qwen2.5-7B-base"]
    for counts, label in zip(counts_list, labels):
        plt.plot(layer_idxs, counts, linestyle="-", label=label)
    plt.xlabel("Layer index",fontweight='bold')
    plt.ylabel(f"Activation Value > {threshold}",fontweight='bold')
    # plt.xticks(layer_idxs)
    ticks = layer_idxs[::4]
    if layer_idxs[-1] not in ticks:
        ticks.append(layer_idxs[-1])
    plt.xticks(ticks)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(prop={'size': 23, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()



def load_int_list(path: str) -> list[int]:
    with open(path, 'r', encoding='utf-8') as f:
        return [int(line.strip()) for line in f if line.strip()]
    

def plot_digit_context_activation(
    pt_path: str,
    neuron_txt: str,
    window_before: int = 50,
    window_after: int = 100,
    output_path: str = "digit_context_activation.png"
):

    data = torch.load(pt_path)
    neuron_ids    = data["neuron_ids"]    
    neuron_layers = data["neuron_layers"]  
    results       = data["results"]        


    selected_neurons = load_int_list(neuron_txt)
    col_idxs = [neuron_ids.index(n) for n in selected_neurons]


    offsets = list(range(-window_before, window_after + 1))
    digits = [str(d) for d in range(10)] + ["Wait"]
    digits = ["Wait","1","the"]


    digit_windows = {d: [] for d in digits}
    for rec in tqdm(results, desc="Scanning prompts"):
        tokens = rec["tokens"]
        acts = rec["activations"][:, col_idxs].cpu().numpy()  
        seq_len = acts.shape[0]

        for pos, tok in enumerate(tokens):
            if tok in digit_windows:
                window_vals = []
                for off in offsets:
                    idx = pos + off
                    if 0 <= idx < seq_len:
                        window_vals.append(acts[idx].mean())
                    else:
                        window_vals.append(np.nan)
                digit_windows[tok].append(window_vals)

    avg_acts = {}
    for d in digits:
        arr = np.array(digit_windows[d], dtype=float)  
        if arr.size:
            avg_acts[d] = np.nanmean(arr, axis=0)
        else:
            avg_acts[d] = np.full(len(offsets), np.nan)

    plt.rcParams['legend.fontsize']    = 27
    plt.rcParams['axes.titlesize']     = 27
    plt.rcParams['axes.labelsize']     = 27
    plt.rcParams['xtick.labelsize']    = 27
    plt.rcParams['ytick.labelsize']    = 27
    plt.figure(figsize=(8, 6))
    for d in digits:
        plt.plot(offsets, avg_acts[d], marker='o', markersize=2, label=f"'{d}'")
    plt.axvline(0, color='gray', linestyle='--', linewidth=2)
    plt.xlabel("Position relative to target token",fontweight='bold')
    plt.ylabel("Avg activation",fontweight='bold')
    plt.legend(prop={'size': 27, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()



def plot_wait_decay_with_log_fit_mutil(
    pt_path: str,
    neuron_txt: str,
    window_before: int = 50,
    window_after: int = 100,
    output_path: str = "wait_context_decay.png"
):
    data = torch.load(pt_path)
    neuron_ids = data["neuron_ids"]
    results    = data["results"]
    selected_neurons = load_int_list(neuron_txt)
    col_idxs = [neuron_ids.index(n) for n in selected_neurons]

    offsets = np.arange(-window_before, window_after + 1)
    digit   = "Wait"
    windows = []

    for rec in tqdm(results, desc="Scanning for 'Wait'"):
        tokens = rec["tokens"]
        acts   = rec["activations"][:, col_idxs].cpu().numpy()  
        L = acts.shape[0]
        for pos, tok in enumerate(tokens):
            if tok == digit:
                vals = [
                    acts[pos + off].mean() if 0 <= pos+off < L else np.nan
                    for off in offsets
                ]
                windows.append(vals)

    arr = np.array(windows, dtype=float)      
    mean_curve = np.nanmean(arr, axis=0)      

    xs = offsets[offsets > 0]
    ys = mean_curve[offsets > 0]

    def decay_func(x, a, b, c):
        return a - b * np.log(x + c)

    p0 = [ys[0], 1.0, 1.0]
    popt, _ = curve_fit(decay_func, xs, ys, p0=p0, maxfev=10000)

    plt.figure(figsize=(10, 6))
    plt.plot(offsets, mean_curve, marker='o', markersize=4, label="Empirical 'Wait'")
    fit_curve = decay_func(xs, *popt)
    plt.plot(xs, fit_curve, color='red', linewidth=2,
             label=f"Fit: a={popt[0]:.3f}, b={popt[1]:.3f}, c={-0.997:.3f}")
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Position relative to 'Wait'")
    plt.ylabel("Avg activation")
    plt.title("Decay of 'Wait' activation and −log fit")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_digit_context_activation(
        pt_path = "/data/LLM_neuron_data/R1_Neuron_All_three_capacity_in_one/R1_activations_multi_gpu/activations_merged.pt",
        neuron_txt = "raasoning_probing_vllm-main_multi_layer_addWait/data/R1_7b_base_all_neurons_200.txt",
        window_before = 50,
        window_after = 100,
        output_path = "/data/LLM_neuron_data/R1_Neuron_All_three_capacity_in_one/R1_token_context_activation.pdf"
    )


def temp_acc(input):
    df = pd.read_csv(input)
    results = []
    verifier_params = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Zero-Shot Inference"):
        results.append({
            "Index": row["Index"],
            "Problem": row["Problem"],
            "Gold Answer": row["Gold Answer"],
            "Prompt": row["Prompt"],
            "Model Answer": row["Model Answer"],
            "Predicted Answer": row["Predicted Answer"],
            "Gold Extracted Answer": row["Gold Extracted Answer"]
        })
        verifier_params.append((row["Index"], row["Predicted Answer"], row["Gold Extracted Answer"]))
    scores = []
    timeout_cnt = 0
    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process_wrapper, verifier_params, timeout=3)
        iterator = future.result()
        for _ in tqdm(range(len(verifier_params)), desc="Evaluating Equivalence"):
            try:
                result = next(iterator)
                scores.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                print(f"TimeoutError for sample: {error}")
                scores.append(False)
                timeout_cnt += 1
            except Exception as error:
                print(f"Error: {error}")
                scores.append(False)
    for i, score in enumerate(scores):
        results[i]["Is Correct"] = score
    df = pd.DataFrame(results)
    accuracy = df["Is Correct"].mean()
    overall_row = pd.DataFrame([{
        "Index": "Overall Accuracy",
        "Problem": "",
        "Gold Answer": "",
        "Prompt": "",
        "Model Answer": "",
        "Predicted Answer": "",
        "Gold Extracted Answer": "",
        "Is Correct": accuracy
    }])
    df = pd.concat([df, overall_row], ignore_index=True)
    print(f"Accuracy: {accuracy:.2%}")