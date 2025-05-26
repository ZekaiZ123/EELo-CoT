import pandas as pd

file1_path = "./result_0.5ref0.75len.csv"


# Load data
df1 = pd.read_csv(file1_path)

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

# Function to compute metrics with word count
def compute_metrics(df):
    # Count words by splitting the text on whitespace
    df['Model Answer Length'] = df['Model Answer'].astype(str).apply(lambda x: len(x.split()))
    df['Self-Reflective'] = df['Model Answer'].apply(contains_self_reflection)
    avg_length = df['Model Answer Length'].mean()
    self_reflection_ratio = df['Self-Reflective'].mean()
    return avg_length, self_reflection_ratio

# Compute metrics for both files
avg_length_1, reflection_ratio_1 = compute_metrics(df1)

# Print results
print("=== Slow Thinking Results ===")
print(f"Average Answer Length: {avg_length_1:.2f} words")
print(f"Self-Reflection Ratio: {reflection_ratio_1:.2%}")
