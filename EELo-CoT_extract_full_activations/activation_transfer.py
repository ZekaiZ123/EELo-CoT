# import os
# import torch
# import pandas as pd
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # All layers patching
# # --- Configuration ---
# GOOD_GROUP = "good_outputs"
# BAD_GROUP = "bad_outputs"
# MODEL_R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_BASE = "Qwen/Qwen2.5-7B"
# CSV_PATH = "filtered_qwen_outputs.csv"
# OUTPUT_DIR = "activation_deltas"
# MAX_LEN = 2048

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# torch.set_grad_enabled(False)

# # --- Helper Functions ---
# def get_act_hook(layer_id, store_dict):
#     def hook(module, input, output):
#         store_dict[layer_id].append(output.detach().cpu())  # [1, seq_len, hidden]
#     return hook

# def insert_hook(layer_id, delta):
#     def patch_fn(module, input, output):
#         return output + delta.to(output.device)
#     return patch_fn

# # --- Load dataset ---
# df = pd.read_csv(CSV_PATH)
# df = df[df['model_output'].notna()]

# # --- Load R1-distilled model for extraction ---
# r1_model = AutoModelForCausalLM.from_pretrained(MODEL_R1, device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer_r1 = AutoTokenizer.from_pretrained(MODEL_R1)

# num_layers = len(r1_model.model.layers)
# good_acts = {l: [] for l in range(num_layers)}
# bad_acts = {l: [] for l in range(num_layers)}

# # --- Register hooks for good_outputs ---
# print("Registering hooks for good outputs...")
# for l in range(num_layers):
#     r1_model.model.layers[l].mlp.act_fn.register_forward_hook(get_act_hook(l, good_acts))

# # --- Run model on good outputs ---
# print("Collecting good activations...")
# for _, row in tqdm(df[df["group"] == GOOD_GROUP].iterrows(), total=(df["group"] == GOOD_GROUP).sum()):
#     input_ids = tokenizer_r1(row["model_output"], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(r1_model.device)
#     _ = r1_model(**input_ids)

# # --- Clear previous hooks, register for bad_outputs ---
# for l in range(num_layers):
#     r1_model.model.layers[l].mlp.act_fn._forward_hooks.clear()
#     r1_model.model.layers[l].mlp.act_fn.register_forward_hook(get_act_hook(l, bad_acts))

# # --- Run model on bad outputs ---
# print("Collecting bad activations...")
# for _, row in tqdm(df[df["group"] == BAD_GROUP].iterrows(), total=(df["group"] == BAD_GROUP).sum()):
#     input_ids = tokenizer_r1(row["model_output"], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(r1_model.device)
#     _ = r1_model(**input_ids)

# # --- Compute deltas ---
# print("Computing activation deltas...")
# deltas = {}
# for l in range(num_layers):
#     if not good_acts[l] or not bad_acts[l]:
#         print(f"Skipping layer {l}: missing activations")
#         continue
#     good_cat = torch.cat(good_acts[l], dim=1)  # [1, total_seq_len, hidden]
#     bad_cat = torch.cat(bad_acts[l], dim=1)
#     mean_good = good_cat.mean(dim=1, keepdim=True)
#     mean_bad = bad_cat.mean(dim=1, keepdim=True)
#     delta = (mean_good - mean_bad).to(torch.bfloat16)
#     deltas[l] = delta
#     torch.save(delta, os.path.join(OUTPUT_DIR, f"delta_layer{l}.pt"))

# # --- Load base model and patch ---
# print("Patching base model with deltas...")
# base_model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer_base = AutoTokenizer.from_pretrained(MODEL_BASE)

# for l, delta in deltas.items():
#     base_model.model.layers[l].mlp.act_fn.register_forward_hook(insert_hook(l, delta))

# # --- Run example inference ---
# prompt = "Let's solve this step-by-step: What is 25% of 80?"
# inputs = tokenizer_base(prompt, return_tensors="pt").to(base_model.device)
# output = base_model.generate(**inputs, max_new_tokens=128)
# print(tokenizer_base.decode(output[0], skip_special_tokens=True))


import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModelForCausalLM

# Load the R1-distilled model temporarily to check num layers
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
num_layers = len(model.model.layers)


# --- Configuration ---
GOOD_GROUP = "good_outputs"
BAD_GROUP = "bad_outputs"
MODEL_R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_BASE = "Qwen/Qwen2.5-7B"
CSV_PATH = "/home/zekai/EELo-CoT_extract_full_activations/filtered_qwen_outputs.csv"
OUTPUT_DIR = "/data/zekai/EELo-CoT_full_activations_extraction/activation_deltas"
MAX_LEN = 4096
TARGET_LAYERS = list(range(max(0, num_layers - 6), num_layers))  # Top 6 layers

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.set_grad_enabled(False)

# --- Helper Functions ---
def get_act_hook(layer_id, store_dict):
    def hook(module, input, output):
        store_dict[layer_id].append(output.detach().cpu())  # [1, seq_len, hidden]
    return hook

def insert_hook(layer_id, delta):
    def patch_fn(module, input, output):
        return output + delta.to(output.device)
    return patch_fn

# --- Load dataset ---
df = pd.read_csv(CSV_PATH)
df = df[df['model_output'].notna()]

# --- Load R1-distilled model for extraction ---
r1_model = AutoModelForCausalLM.from_pretrained(MODEL_R1, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer_r1 = AutoTokenizer.from_pretrained(MODEL_R1)

good_acts = {l: [] for l in TARGET_LAYERS}
bad_acts = {l: [] for l in TARGET_LAYERS}

# --- Register hooks for good_outputs ---
print("Registering hooks for good outputs...")
for l in TARGET_LAYERS:
    r1_model.model.layers[l].mlp.act_fn.register_forward_hook(get_act_hook(l, good_acts))

# --- Run model on good outputs ---
print("Collecting good activations...")
for _, row in tqdm(df[df["group"] == GOOD_GROUP].iterrows(), total=(df["group"] == GOOD_GROUP).sum()):
    input_ids = tokenizer_r1(row["model_output"], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(r1_model.device)
    _ = r1_model(**input_ids)

# --- Clear hooks and register for bad_outputs ---
for l in TARGET_LAYERS:
    r1_model.model.layers[l].mlp.act_fn._forward_hooks.clear()
    r1_model.model.layers[l].mlp.act_fn.register_forward_hook(get_act_hook(l, bad_acts))

# --- Run model on bad outputs ---
print("Collecting bad activations...")
for _, row in tqdm(df[df["group"] == BAD_GROUP].iterrows(), total=(df["group"] == BAD_GROUP).sum()):
    input_ids = tokenizer_r1(row["model_output"], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(r1_model.device)
    _ = r1_model(**input_ids)

# --- Compute deltas ---
print("Computing activation deltas...")
deltas = {}
for l in TARGET_LAYERS:
    if not good_acts[l] or not bad_acts[l]:
        print(f"Skipping layer {l}: missing activations")
        continue
    good_cat = torch.cat(good_acts[l], dim=1)
    bad_cat = torch.cat(bad_acts[l], dim=1)
    mean_good = good_cat.mean(dim=1, keepdim=True)
    mean_bad = bad_cat.mean(dim=1, keepdim=True)
    delta = (mean_good - mean_bad).to(torch.bfloat16)
    deltas[l] = delta
    torch.save(delta, os.path.join(OUTPUT_DIR, f"delta_layer{l}.pt"))

# --- Load base model and apply patches ---
print("Patching base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer_base = AutoTokenizer.from_pretrained(MODEL_BASE)

for l, delta in deltas.items():
    base_model.model.layers[l].mlp.act_fn.register_forward_hook(insert_hook(l, delta))

# --- Inference example ---
example = "Let's carefully compute: What is 15% of 240?"
inputs = tokenizer_base(example, return_tensors="pt").to(base_model.device)
output = base_model.generate(**inputs, max_new_tokens=128)
print(tokenizer_base.decode(output[0], skip_special_tokens=True))
