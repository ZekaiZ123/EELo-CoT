import pandas as pd
import os
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import re
# Import both math_equal_process_wrapper and extract_answer from parser
from parser import math_equal_process_wrapper, extract_answer

# Define self-reflection phrases
reflection_keywords = [
    "let me double check", "wait", "i think", "on second thought",
    "perhaps", "actually", "maybe", "i realize", "i should",
    "it seems", "i believe", "i thought", "it might be", "i guess",
    "verify", "double-check", "reconsider", "double check",
    "check again", "ensure", "confirm", "look back"
]

# Helper function to detect self-reflective phrases
def contains_self_reflection(text):
    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in reflection_keywords)

# Function to get text before "boxed" keyword
def get_text_before_boxed(text):
    text = str(text)
    text = re.split(r"\n\s*(?:Human|Assistant)\s*:", text)[0]
    return text

def load_and_evaluate(interim_csv_path, output_csv_path, data_name="math"):
    """
    Load intermediate results and perform evaluation
    
    Args:
        interim_csv_path: Path to the intermediate CSV file
        output_csv_path: Path to save the final results with evaluation
        data_name: Name of the dataset (default: "math")
    """
    # Load the intermediate results
    print(f"Loading intermediate results from {interim_csv_path}")
    df1 = pd.read_csv(interim_csv_path)
    
    # Prepare verifier parameters and re-extract answers
    all_results = df1.to_dict('records')
    
    print("Re-extracting answers from model responses...")
    for item in tqdm(all_results, desc="Extracting Answers"):
        # Extract answer from the model's raw response
        raw_answer = item["Model Answer"]
        item["Re-extracted Answer"] = extract_answer(raw_answer, data_name)
        
        # Get text before "boxed" keyword for metrics
        text_before_boxed = get_text_before_boxed(raw_answer)
        item["Text Before Boxed"] = text_before_boxed
        item["Word Count"] = len(text_before_boxed.split())
        item["Self-Reflective"] = contains_self_reflection(text_before_boxed)
    
    all_verifier_params = [(item["Index"], item["Re-extracted Answer"], str(item["Gold Extracted Answer"])) 
                           for item in all_results]
    
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
    
    # Calculate metrics
    avg_word_count = df["Word Count"].mean()
    self_reflection_ratio = df["Self-Reflective"].mean()
    
    overall_row = pd.DataFrame([{
        "Index": "Overall Accuracy",
        "Problem": "",
        "Gold Answer": "",
        "Prompt": "",
        "Model Answer": "",
        "Predicted Answer": "",
        "Re-extracted Answer": "",
        "Gold Extracted Answer": "",
        "Is Correct": accuracy,
        "Word Count": avg_word_count,
        "Self-Reflective": self_reflection_ratio
    }])
    
    df = pd.concat([df, overall_row], ignore_index=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}. Accuracy: {accuracy:.2%}")
    print(f"Average Word Count (before 'boxed'): {avg_word_count:.2f} words")
    print(f"Self-Reflection Ratio (before 'boxed'): {self_reflection_ratio:.2%}")
    
    return accuracy, avg_word_count, self_reflection_ratio

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and evaluate interim results")
    parser.add_argument("--interim_file", type=str, required=True, 
                        help="Path to the interim CSV file")
    
    args = parser.parse_args()
    output_file = args.interim_file.split(".")[0] + "_with_metrics.csv"
    load_and_evaluate(args.interim_file, output_file, "math") 