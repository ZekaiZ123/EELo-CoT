import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm

# tell PyTorch to use expandable segments for CUDA allocations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def extract_reasoning_only(text):
    return text.split("assistant\n", 1)[-1].strip() if "assistant\n" in text else text.strip()



hook_activations = {}
current_reflection_range = (0, None)


def get_mlp_hook(module_key):

    def hook(module, inputs, output):
        global hook_activations, current_reflection_range
        start_idx, end_idx = current_reflection_range
        if end_idx is None:
            end_idx = output.size(1)

        max_activation, _ = output[:, start_idx:end_idx, :].max(dim=1)
        hook_activations[module_key] = max_activation.squeeze(0).detach().cpu()

    return hook


def register_mlp_hooks(model, layer_id):

    mlp_modules = [(name, module) for name, module in model.named_modules()
                   if "mlp.act_fn" in name]
    if not mlp_modules:
        raise ValueError("No MLP or feed_forward modules found in the model.")

    # Register hook on the last MLP module
    module_key, module = mlp_modules[layer_id]
    module.register_forward_hook(get_mlp_hook(module_key))
    print(f"Registered MLP hook on module: {module_key}")


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-7B"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, max_length=10240)
    return model, tokenizer



def get_mean_mlp_activations(text, model, tokenizer):

    global hook_activations, current_reflection_range

    reasoning_text = extract_reasoning_only(text)
    encoding = tokenizer(reasoning_text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=10240)
    total_tokens = encoding["input_ids"].size(1)
    current_reflection_range = (0, total_tokens)
    hook_activations = {}  # Reset stored activations

    inputs = tokenizer(reasoning_text, return_tensors="pt", truncation=True, max_length=10240)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        _ = model(**inputs)

    return hook_activations



def process_dataset(csv_path, model, tokenizer):
    df = pd.read_csv(csv_path)
    df = df[df['model_output'].notna()]
    good_activations = []
    bad_activations = []

    # for i, row in df.iterrows():
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Scanning prompts"):
        text = row["model_output"]
        group = row["group"].strip().lower()
        activations = get_mean_mlp_activations(text, model, tokenizer)
        if activations:
            act = list(activations.values())[0]
            if group == "good_outputs":
                good_activations.append(act)
            elif group == "bad_outputs":
                bad_activations.append(act)

    if not good_activations or not bad_activations:
        raise ValueError("Insufficient data for good or bad cases.")

    good_tensor = torch.stack(good_activations)
    bad_tensor = torch.stack(bad_activations)

    mean_good = good_tensor.mean(dim=0)
    mean_bad = bad_tensor.mean(dim=0)
    delta = mean_good - mean_bad
    return delta, mean_good, mean_bad


# def Acc_neuron(model_name, output, layer_id):
def Acc_neuron(gpu_id: int, model_name: str, layer_id: int, output: str, csv_path: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    #  = "Qwen/Qwen2.5-Math-7B-instruct"  # Change if needed
    

    model, tokenizer = load_model_and_tokenizer(model_name)

    register_mlp_hooks(model, layer_id)
    delta, max_good, max_bad = process_dataset(csv_path, model, tokenizer)
    top_k = 1000
    top_values, top_indices = torch.topk(delta, k=top_k)


    results = []
    print("Top neurons that support good reflection (clean supportive neurons) in the last MLP layer:")
    for idx, delta_val in zip(top_indices.tolist(), top_values.tolist()):
        results.append({
            "Neuron_ID": idx,
            "act_val": delta_val,
            "Category": "accuracy",
        })
    df = pd.DataFrame(results)
    df.to_csv(f"{output}/math500_acc{layer_id}_2000.csv", index=False)
    return f"{output}/math500_acc{layer_id}_2000.csv"

from concurrent.futures import ProcessPoolExecutor

def main():
    model_name = "Qwen/Qwen2.5-7B"
    output = "/data/LLM_neuron_data/base"
    csv_path = "selected_data_pairs.csv"

    # 要跑的 layer_id 列表
    layer_ids = list(range(28))  # 举例：0–27
    num_gpus  = torch.cuda.device_count()

    # 用 ProcessPoolExecutor 简单分配任务
    tasks = []
    with ProcessPoolExecutor(max_workers=num_gpus) as exe:
        for i, layer_id in enumerate(layer_ids):
            gpu_id = i % num_gpus
            tasks.append(
                exe.submit(Acc_neuron, gpu_id, model_name, layer_id, output, csv_path)
            )


if __name__ == "__main__":
    main()

