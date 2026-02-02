import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
from collections import Counter


# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Suppress warnings and progress bars
    warnings.filterwarnings("ignore")
    pd.options.mode.chained_assignment = None

    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    print("=== STARTING EDA ===")

    # --------------------------------------------------------------------------
    # 1. Data Loading & Integrity
    # --------------------------------------------------------------------------
    # Load Metadata (Defines the Training Set)
    try:
        meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    except FileNotFoundError:
        print("Error: Metadata file not found.")
        return

    # Load Text Data
    # We only read 'id' and 'comment_text' from the source to save memory/time
    # The metadata contains the labels and split info.
    source_train_path = os.path.join(INPUT_DIR, "train.csv")

    # Read full train file (cols: id, comment_text)
    # Note: The original train.csv has all columns, but we only need text to merge.
    # We use the metadata's 'target' as the ground truth.
    text_df = pd.read_csv(source_train_path, usecols=["id", "comment_text"])

    # Merge: Inner join ensures we only analyze the specific rows allocated to 'train'
    # by the metadata generation step.
    df = meta_train.merge(text_df, on="id", how="inner")

    # Fill NaNs in text with empty string just in case
    df["comment_text"] = df["comment_text"].fillna("")

    print(f"Data Loaded. Training Samples: {len(df)}")

    # --------------------------------------------------------------------------
    # 2. Target Variable Analysis
    # --------------------------------------------------------------------------
    print("\nTARGET VARIABLE ANALYSIS")

    target_col = "target"
    targets = df[target_col]

    # Continuous Analysis
    print(f"Target Type: Continuous Fraction [0, 1]")
    print(f"Mean: {targets.mean():.4f}")
    print(f"Std Dev: {targets.std():.4f}")
    print(f"Min: {targets.min():.4f}")
    print(f"Max: {targets.max():.4f}")

    # Skewness and Kurtosis
    print(f"Skewness: {targets.skew():.4f}")
    print(f"Kurtosis: {targets.kurtosis():.4f}")

    # Binary Classification Analysis (Threshold >= 0.5)
    binary_targets = (targets >= 0.5).astype(int)
    pos_count = binary_targets.sum()
    neg_count = len(binary_targets) - pos_count
    total = len(binary_targets)
    pos_ratio = pos_count / total

    print(f"Binary Class Balance (Threshold 0.5):")
    print(f"  Toxic (1): {pos_count} ({pos_ratio:.4f})")
    print(f"  Non-Toxic (0): {neg_count} ({1 - pos_ratio:.4f})")
    print(f"  Ratio (Neg/Pos): {neg_count/pos_count:.4f}")

    # --------------------------------------------------------------------------
    # 3. Input Data Analysis (Text Modality)
    # --------------------------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # Compute Lengths
    # Character Length
    char_lens = df["comment_text"].str.len()

    # Word Length (Simple whitespace split)
    # We use a sample for very expensive operations if needed, but vectorization is fast enough here.
    word_lens = df["comment_text"].str.split().str.len()

    print("Sequence Lengths (Characters):")
    print(f"  Mean: {char_lens.mean():.4f}")
    print(f"  Std:  {char_lens.std():.4f}")
    print(f"  Min:  {char_lens.min():.4f}")
    print(f"  Max:  {char_lens.max():.4f}")

    print("Sequence Lengths (Words):")
    print(f"  Mean: {word_lens.mean():.4f}")
    print(f"  Std:  {word_lens.std():.4f}")
    print(f"  Min:  {word_lens.min():.4f}")
    print(f"  Max:  {word_lens.max():.4f}")

    # Vocabulary Analysis
    # To keep runtime low, we analyze a random sample of 100k rows for vocabulary stats
    sample_size = min(100000, len(df))
    vocab_sample = df["comment_text"].sample(n=sample_size, random_state=42)

    # Basic tokenization
    all_tokens = [word for text in vocab_sample for word in text.split()]
    token_counts = Counter(all_tokens)

    vocab_size = len(token_counts)
    total_tokens = len(all_tokens)

    # Calculate OOV potential (tokens appearing only once in the sample)
    rare_tokens = sum(1 for count in token_counts.values() if count == 1)

    print(f"Vocabulary Statistics (Sample N={sample_size}):")
    print(f"  Unique Vocabulary Size: {vocab_size}")
    print(f"  Total Tokens in Sample: {total_tokens}")
    print(
        f"  Rare Tokens (Freq=1): {rare_tokens} ({rare_tokens/vocab_size:.4f} of Vocab)"
    )
    print(f"  Avg Tokens per Sample: {total_tokens/sample_size:.4f}")

    # --------------------------------------------------------------------------
    # 4. Input Data Analysis (Tabular/Identity Attributes)
    # --------------------------------------------------------------------------
    # Although the primary modality is text, the identity columns are critical inputs
    # for the bias mitigation task.
    print("\nINPUT DATA ANALYSIS (IDENTITY ATTRIBUTES)")

    identity_cols = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]

    # Check for presence of identity columns
    available_ids = [c for c in identity_cols if c in df.columns]

    if available_ids:
        # Calculate sparsity (percentage of rows where identity > 0)
        print("Identity Mention Frequency (Non-zero entries):")
        for col in available_ids:
            non_zero = (df[col] > 0).mean()
            print(f"  {col}: {non_zero:.4f}")

        # Check for missing values in these columns
        na_counts = df[available_ids].isna().mean()
        avg_na = na_counts.mean()
        print(f"Average Missing Value Rate in Identity Cols: {avg_na:.4f}")
    else:
        print("No identity columns found in metadata.")

    # --------------------------------------------------------------------------
    # 5. Feature/Signal Relationships
    # --------------------------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Structured Relationships: Correlation
    # We correlate Target with Identity Attributes and Toxicity Subtypes
    print("Correlations with Target (Top 5 Positive):")

    # Select numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(df["target"]).sort_values(ascending=False)

    # Filter out the target itself
    correlations = correlations.drop(labels=["target"], errors="ignore")

    for name, val in correlations.head(5).items():
        print(f"  {name}: {val:.4f}")

    # Unstructured (Meta-Feature) Relationships
    # Correlation between Text Length and Target
    # We create a temporary dataframe for this calculation
    meta_features = pd.DataFrame(
        {"target": df["target"], "char_length": char_lens, "word_length": word_lens}
    )

    len_corr = meta_features.corr().loc["target"]

    print("Meta-Feature Correlations with Target:")
    print(f"  Character Length: {len_corr['char_length']:.4f}")
    print(f"  Word Length:      {len_corr['word_length']:.4f}")

    # Check if longer comments are more likely to be toxic
    # Compare mean length of Toxic vs Non-Toxic
    toxic_mask = df["target"] >= 0.5
    avg_len_toxic = word_lens[toxic_mask].mean()
    avg_len_nontoxic = word_lens[~toxic_mask].mean()

    print("Average Word Length by Class:")
    print(f"  Toxic:     {avg_len_toxic:.4f}")
    print(f"  Non-Toxic: {avg_len_nontoxic:.4f}")


if __name__ == "__main__":
    main()
