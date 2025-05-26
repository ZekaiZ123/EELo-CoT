import torch
import pandas as pd
import torch.multiprocessing as mp
from functools import partial
import os
import json

from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import random
from parser import strip_string


import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import os
import argparse
import glob
from parser import parse_ground_truth, extract_answer, math_equal_process_wrapper
from model import DeepSeekQwenModel
from transformers import AutoTokenizer, Qwen2ForCausalLM
from load_and_evaluate import load_and_evaluate
from intervene_functions import (
    ConstantInterveneFunction,
    KeywordInterveneFunction,
    ConstantDecayInterveneFunction,
    KeywordDecayInterveneFunction
)


def generate_answer(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.6, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

def build_zero_shot_prompt(question):
    return (
        "What is the correct answer to this question:\n"
        f"### Problem:\n{question}\n\n"
        "Reason through your answer step-by-step. Then, based on your reasoning, provide the single most likely answer choice. Answer in the format: \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}"
    )

def initialize_intervene_functions(config, tokenizer=None, modules=None):
    """Initialize intervention functions based on config
    
    Args:
        config: Dict or path to JSON config file containing intervention settings
        tokenizer: Optional tokenizer to convert token strings to IDs
        
    Returns:
        List of initialized intervention functions
    """
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    
    intervene_functions = []
    
    for func_config in config["intervene_functions"]:
        func_type = func_config["type"]
        
        # Load top_neurons from file if path is provided
        if "top_neurons_file" in func_config:
            with open(func_config["top_neurons_file"], 'r') as f:
                top_neurons = [int(line.strip()) for line in f.readlines() if line.strip()]
            func_config["top_neurons"] = top_neurons

        if "layer_list_file" in func_config:
            with open(func_config["layer_list_file"], 'r') as f:
                layer_list = [int(line.strip()) for line in f.readlines() if line.strip()]
            func_config["layer_list"] = layer_list
        
        # Load keywords from file if path is provided
        if "keywords_file" in func_config:
            with open(func_config["keywords_file"], 'r') as f:
                keywords = [line.strip() for line in f.readlines() if line.strip()]
            
            # Convert token strings to token IDs if tokenizer is provided
            if tokenizer is not None and all(isinstance(k, str) for k in keywords):
                keywords = tokenizer.convert_tokens_to_ids(keywords)
            func_config["keywords"] = keywords
        # Process keywords if present in the config
        elif "keywords" in func_config and tokenizer is not None and isinstance(func_config["keywords"][0], str):
            # Convert token strings to token IDs
            func_config["keywords"] = tokenizer.convert_tokens_to_ids(func_config["keywords"])
        
        # Initialize the appropriate function based on type
        if func_type == "ConstantIntervene":
            intervene_function = ConstantInterveneFunction(
                amp=func_config["amp"],
                neuron_list=func_config["top_neurons"],
                n_neurons=func_config["n_neurons"],
                layer_list=func_config["layer_list"]
            )
        elif func_type == "KeywordIntervene":
            intervene_function = KeywordInterveneFunction(
                amp=func_config["amp"],
                top_neurons=func_config["top_neurons"],
                n_neurons=func_config["n_neurons"],
                keywords=func_config["keywords"]
            )
        elif func_type == "ConstantDecayIntervene":
            intervene_function = ConstantDecayInterveneFunction(
                amp=func_config["amp"],
                top_neurons=func_config["top_neurons"],
                n_neurons=func_config["n_neurons"],
                t_max=func_config["t_max"]
            )
        elif func_type == "KeywordDecayIntervene":
            intervene_function = KeywordDecayInterveneFunction(
                amp=func_config["amp"],
                top_neurons=func_config["top_neurons"],
                n_neurons=func_config["n_neurons"],
                layer_list=func_config["layer_list"],
                t_max=func_config["t_max"],
                keywords=func_config["keywords"],
                t_initial=func_config["t_initial"],
                cool_down=func_config["cool_down"],
                modules        = modules,
                tokenizer=tokenizer
            )
        else:
            raise ValueError(f"Unknown intervention function type: {func_type}")
        
        intervene_functions.append(intervene_function)
    
    return intervene_functions

def process_chunk(gpu_id, data_chunk, config_path, data_name="math"):
    """Process a chunk of the dataset on a specific GPU using config-based intervention"""
    device = f"cuda:{gpu_id}"
    
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)


    # Initialize model on this GPU
    model_id = config.get("model", "Qwen/Qwen2.5-Math-7B")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Initialize intervention functions from config
    intervene_functions = initialize_intervene_functions(config, tokenizer)
    
    
    # First load the model without intervention functions
    deepseek_qwen = DeepSeekQwenModel.from_pretrained(model_id, intervene_functions=intervene_functions, layer_idx=config.get("layer_idx", 27))

    
    deepseek_qwen.to(device)
    deepseek_qwen.eval()
    deepseek_qwen.tokenizer = tokenizer
    deepseek_qwen.generation_config.pad_token_id = deepseek_qwen.tokenizer.pad_token_id
    
    # Set generation parameters from config if available
    if "generation_params" in config:
        generation_params = config["generation_params"]
    else:
        generation_params = {
            "max_new_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.9,
            "_sentence_cooldown": 4
        }

    deepseek_qwen._sentence_cooldown = generation_params.get("_sentence_cooldown", 4)
    
    
    results = []
    verifier_params = []
    
    print(f"GPU {gpu_id}: Processing {len(data_chunk)} examples")
    
    for i, item in enumerate(tqdm(data_chunk, desc=f"GPU {gpu_id} Inference", position=gpu_id)):
        question = item["problem"]
        question = question.split("Please write your final answer")[0]
        # _, gold_extracted = parse_ground_truth(item, data_name)
        gold_extracted = extract_answer(item["solution"], data_name)
        prompt = build_zero_shot_prompt(question)
        raw_answer = generate_answer(
            deepseek_qwen, 
            deepseek_qwen.tokenizer, 
            prompt=prompt, 
            max_new_tokens=generation_params.get("max_new_tokens", 2048),
            temperature=generation_params.get("temperature", 0.6),
            top_p=generation_params.get("top_p", 0.9)
        )
        pred_extracted = extract_answer(raw_answer, data_name)
        
        # Store the original index for later recombination
        original_idx = item.get("__original_idx", i)
        
        results.append({
            "Index": original_idx,
            "Problem": question,
            "Gold Answer": item["solution"],
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
def process_wrapper(args_config):
    args, config_path = args_config
    gpu_id, chunk = args
    return process_chunk(gpu_id, chunk, config_path)

def evaluate_model_on_math500_parallel(test_data, config_path, num_gpus=4, 
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
    args_with_data = [((i, chunks[i]), config_path) for i in range(num_gpus)]
    
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
    
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}. Accuracy: {accuracy:.2%}")
    accuracy, avg_word_count, self_reflection_ratio = load_and_evaluate(output_csv, output_csv, "math")
    return accuracy, avg_word_count, self_reflection_ratio

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run inference with model intervention")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to a single config file or directory of config files")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs to use for parallel inference")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    args = parser.parse_args()
    
    # Set multiprocessing method
    mp.set_start_method('spawn', force=True)
    
    # Load the MATH-500 dataset (test split)
    math500_dataset = load_dataset("hendrydong/gpqa_diamond_mc")
    test_data = math500_dataset["test"]
    # test_data = test_data[:100]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if the config path is a file or directory
    if os.path.isfile(args.config):
        # Single config file
        config_paths = [args.config]
    else:
        # Directory of config files
        config_paths = glob.glob(os.path.join(args.config, "*.json"))
    
    if not config_paths:
        raise ValueError(f"No config files found at {args.config}")
    
    print(config_paths)
    
    # Use parallel evaluation with specified number of GPUs
    for i, config_path in enumerate(config_paths):
        print(f"Processing config {i+1}/{len(config_paths)}: {config_path}")
        
        # Extract config filename for result naming
        config_filename = os.path.basename(config_path)
        output_csv = os.path.join(args.output_dir, f"result_from_{config_filename}.csv")
        output_txt = os.path.join(args.output_dir, f"result_from_{config_filename}.txt")
        
        accuracy, avg_word_count, self_reflection_ratio = evaluate_model_on_math500_parallel(
            test_data,
            config_path=config_path,
            num_gpus=args.num_gpus,
            output_csv=output_csv
        )
        
        with open(output_txt, "a") as f:
            f.write(f"Accuracy: {accuracy:.2%}, Avg Word Count: {avg_word_count:.2f}, Self Reflection Ratio: {self_reflection_ratio:.2%}\n")
        
        print(f"Results saved to {output_csv} and {output_txt}")
