import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from vllm import LLM, SamplingParams
import torch
from vllm import ModelRegistry
from qwen_vllm import Qwen2ProbingForCausalLM

ModelRegistry.register_model("Qwen2ProbingForCausalLM", Qwen2ProbingForCausalLM)

def build_zero_shot_prompt(question):
    return (
        "Solve the problem by reasoning step by step"
        "before providing the final answer. Explain each step clearly."
        "Finally, provide your final answer \n\n"
        "in LaTeX format: \\boxed{Your answer}"

        f"### Problem:\n{question}\n\n"
        "### Step-by-Step Solution:\n"
        "Let's think step by step:\n\n"
    )


def main(args):
    # ---------- load data ----------
    df = pd.read_csv(args.input_csv)
    if "problem" not in df.columns:
        raise ValueError("CSV must contain a 'problem' column")
    df = df[:4]
    # take first 16 for testing
    llm = LLM(
        model="/home/anananan116/reasoning_probing/qwen7b_prob",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.gpu_memory / 100,
        tensor_parallel_size=args.tp
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=8192
    )

    # ---------- inference ----------
    prompts = [build_zero_shot_prompt(q) for q in df["problem"].tolist()]
    for j in range(1):
        print(f"Starting inference {j}")
        predictions = []
        
        for i in tqdm(range(0, len(prompts), args.batch_size),
                    desc="Generating"):
            batch_prompts = prompts[i : i + args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            # vLLM returns Obj with .outputs[0].text
            for out in outputs:
                pred = out.outputs[0].text.strip()
                predictions.append(pred)

        df[f"prediction_{j}"] = predictions

    # ---------- save ----------
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, default="train.csv",
                   help="CSV containing the problems")
    p.add_argument("--output_csv", type=str, default="result_7b.csv",
                   help="Where to write the CSV with predictions")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Number of prompts per forward pass")
    p.add_argument("--gpu_memory", type=float, default=80,
                   help="Percentage (0‑100) of each GPU’s memory vLLM may use")
    p.add_argument("--tp", type=int, default=2,
                   help="Tensor parallel degree (num GPUs to shard across)")
    args = p.parse_args()
    main(args)
