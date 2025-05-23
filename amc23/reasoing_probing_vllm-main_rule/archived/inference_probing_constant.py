import torch
import pandas as pd
import torch.multiprocessing as mp
from functools import partial
import os

from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import random

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import os
from parser import parse_ground_truth, extract_answer, math_equal_process_wrapper
from model_constant import DeepSeekQwenModel
from transformers import AutoTokenizer
from load_and_evaluate import load_and_evaluate
from transformers import Qwen2ForCausalLM
from intervene_functions import ConstantInterveneFunction, ConstantDecayInterveneFunction, KeywordInterveneFunction, KeywordDecayInterveneFunction

def register_amplification_hook(model, layer_idx: int, top_neurons: list, factor: float):
    try:
        layer = model.model.layers[layer_idx]
        target_module = layer.mlp.act_fn
        target_module_name = f"model.model.layers[{layer_idx}].mlp.act_fn"
    except (AttributeError, IndexError) as e:
        raise ValueError(
            f"Could not find target module for amplification at layer index {layer_idx}. "
            f"Please verify the model architecture. Error: {e}"
        ) from e

    def amplification_hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Output is not a tensor.")
        if output.shape[-1] <= max(top_neurons):
            raise IndexError("Specified top_neurons indices are out of range for the output tensor.")
        output[..., top_neurons] *= factor
        return output

    amplification_handle = target_module.register_forward_hook(amplification_hook)
    print(f"Registered amplification hook on module: {target_module_name} with factor {factor}")

def generate_answer(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.6, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = len(inputs["input_ids"][0])
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    raw_output = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
    return raw_output

def build_zero_shot_prompt(question):
    return (
        f"Human: {question}\n\n"
        "Assistant: To solve this problem accurately, I will provide a detailed step-by-step solution:\n"
    )

def process_chunk(gpu_id, data_chunk, ref_scale_dict, acc_neuron, len_neuron, data_name="math"):
    """Process a chunk of the dataset on a specific GPU"""
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
    acc_table = pd.read_csv("token_ratio_acc.csv")
    acc_tokens = []
    for row in acc_table.iterrows():
        acc_tokens.append(row[1]["token"])
    acc_tokens = acc_tokens[:300]
    # Convert token strings to token IDs
    acc_token_ids = tokenizer.convert_tokens_to_ids(acc_tokens)
    
    # Initialize model on this GPU
    deepseek_qwen = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B")
    register_amplification_hook(deepseek_qwen, 27, acc_neuron, 1.2)
    deepseek_qwen.to(device)
    deepseek_qwen.eval()
    deepseek_qwen.generation_config.pad_token_id = tokenizer.pad_token_id
    results = []
    verifier_params = []
    
    print(f"GPU {gpu_id}: Processing {len(data_chunk)} examples")
    
    for i, item in enumerate(tqdm(data_chunk, desc=f"GPU {gpu_id} Inference", position=gpu_id)):
        question = item["problem"]
        _, gold_extracted = parse_ground_truth(item, data_name)
        prompt = build_zero_shot_prompt(question)
        raw_answer = generate_answer(deepseek_qwen, tokenizer, prompt=prompt, max_new_tokens=2048)
        pred_extracted = extract_answer(raw_answer, data_name)
        
        # Store the original index for later recombination
        original_idx = item.get("__original_idx", i)
        
        results.append({
            "Index": original_idx,
            "Problem": question,
            "Gold Answer": item["answer"],
            "Prompt": prompt,
            "Model Answer": raw_answer,
            "Predicted Answer": pred_extracted,
            "Gold Extracted Answer": gold_extracted
        })
        verifier_params.append((original_idx, pred_extracted, gold_extracted))
    
    # Free GPU memory
    del deepseek_qwen
    torch.cuda.empty_cache()
    
    return results, verifier_params

# Define process_wrapper outside of any function to make it picklable
def process_wrapper(args_ref_scale_acc_len):
    args, ref_scale_dict, acc_neuron, len_neuron = args_ref_scale_acc_len
    gpu_id, chunk = args
    return process_chunk(gpu_id, chunk, ref_scale_dict, acc_neuron, len_neuron)

def evaluate_model_on_math500_parallel(test_data, ref_scale_dict, acc_neuron, len_neuron, num_gpus=8, 
                                      output_csv="math500_results_zero_shot.csv"):
    # Add original indices to dataset for recombining later
    test_data_with_idx = []
    for i, item in enumerate(test_data):
        item_copy = dict(item)
        item_copy["__original_idx"] = i
        test_data_with_idx.append(item_copy)
    
    # Split data into chunks for each GPU
    chunk_size = len(test_data_with_idx) // num_gpus
    remainder = len(test_data_with_idx) % num_gpus
    
    chunks = []
    start_idx = 0
    for i in range(num_gpus):
        # Add one extra item to some chunks if division isn't even
        extra = 1 if i < remainder else 0
        end_idx = start_idx + chunk_size + extra
        chunks.append(test_data_with_idx[start_idx:end_idx])
        start_idx = end_idx
    
    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Launch processes for each GPU using the external process_wrapper function
    args_with_data = [((i, chunks[i]), ref_scale_dict, acc_neuron, len_neuron) for i in range(num_gpus)]
    
    with mp.Pool(num_gpus) as pool:
        results = pool.map(process_wrapper, args_with_data)
    
    # Combine results from all GPUs
    all_results = []
    all_verifier_params = []
    
    for chunk_results, chunk_verifier_params in results:
        all_results.extend(chunk_results)
        all_verifier_params.extend(chunk_verifier_params)
    
    # Sort results by original index
    all_results.sort(key=lambda x: x["Index"])
    all_verifier_params.sort(key=lambda x: x[0])
    
    # Save intermediate results
    df1 = pd.DataFrame(all_results)
    df1.to_csv(f"{output_csv}.interim", index=False)
    
    # Verify answers
    scores = []
    timeout_cnt = 0
    with ProcessPool(max_workers=os.cpu_count()) as pool:
        future = pool.map(math_equal_process_wrapper, all_verifier_params, timeout=3)
        iterator = future.result()
        for _ in tqdm(range(len(all_verifier_params)), desc="Evaluating Equivalence"):
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
    
    # Add correctness to results
    for i, score in enumerate(scores):
        all_results[i]["Is Correct"] = score
    
    df = pd.DataFrame(all_results)
    accuracy = df["Is Correct"].mean()
    
    # overall_row = pd.DataFrame([{
    #     "Index": "Overall Accuracy",
    #     "Problem": "",
    #     "Gold Answer": "",
    #     "Prompt": "",
    #     "Model Answer": "",
    #     "Predicted Answer": "",
    #     "Gold Extracted Answer": "",
    #     "Is Correct": accuracy
    # }])
    
    # df = pd.concat([df, overall_row], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}. Accuracy: {accuracy:.2%}")
    load_and_evaluate(output_csv, output_csv, "math")
    return all_results, df

if __name__ == "__main__":
    # Set multiprocessing method
    mp.set_start_method('spawn', force=True)
    
    # Load the MATH-500 dataset (test split)
    math500_dataset = load_dataset("HuggingFaceH4/MATH-500")
    test_data = math500_dataset["test"]
    
    # Load neuron IDs from your combined 100 neurons CSV
    combined_df_ref = pd.read_csv("from670_on_math500_100_total_acc-ref-len.csv")

    total_neuron = combined_df_ref["Neuron_ID"].tolist()
   
    acc_neuron = total_neuron
    ref_neuron = total_neuron[34:67]
    len_neuron = total_neuron[67:100]
    
    ref_scale_list = [2.0 - (1.0/len(ref_neuron))*i for i in range(len(ref_neuron))]
    ref_scale_dict = dict(zip(ref_neuron, ref_scale_list))

    # Use parallel evaluation with 8 GPUs
    results, df = evaluate_model_on_math500_parallel(
        test_data,
        ref_scale_dict=ref_scale_dict,
        acc_neuron=acc_neuron,
        len_neuron=len_neuron,
        num_gpus=8,
        output_csv="./results/result_argmax_constant_noprompt_34_33_33_neurons_1.2_1.2_1.2.csv"
    )
