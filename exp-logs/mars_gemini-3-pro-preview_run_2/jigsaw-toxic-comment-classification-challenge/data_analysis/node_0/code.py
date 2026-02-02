import pandas as pd
import numpy as np
import os
import sys
import random
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Configuration
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    LABEL_COLS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    set_seed(42)

    print("=== STARTING EXPLORATORY DATA ANALYSIS ===\n")

    # 1. Data Integrity
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    # Load data
    # Using engine='c' for performance
    df = pd.read_csv(TRAIN_PATH)

    # Handle potential missing text values
    df["comment_text"] = df["comment_text"].fillna("")

    print("=== DATASET OVERVIEW ===")
    print(f"Total Training Samples: {len(df)}")
    print(f"Number of Target Labels: {len(LABEL_COLS)}\n")

    # 2. Target Variable Analysis
    print("=== TARGET VARIABLE ANALYSIS ===")

    # Class Balance / Distribution
    print("--- Class Distribution ---")
    print(f"{'Label':<20} {'Count':<10} {'Percentage':<10}")

    total_samples = len(df)
    for label in LABEL_COLS:
        count = df[label].sum()
        percentage = (count / total_samples) * 100
        print(f"{label:<20} {count:<10} {percentage:.4f}%")

    # Multi-label analysis
    print("\n--- Multi-Label Cardinality ---")
    # Sum across rows to see how many labels each comment has
    df["label_count"] = df[LABEL_COLS].sum(axis=1)
    label_counts = df["label_count"].value_counts().sort_index()

    print(f"{'Labels per Comment':<20} {'Count':<10} {'Percentage':<10}")
    for num_labels, count in label_counts.items():
        pct = (count / total_samples) * 100
        print(f"{num_labels:<20} {count:<10} {pct:.4f}%")

    # Check for clean vs toxic split
    clean_count = label_counts.get(0, 0)
    clean_pct = (clean_count / total_samples) * 100
    print(f"\nClean Comments (0 labels): {clean_count} ({clean_pct:.4f}%)")
    print(
        f"Toxic Comments (>=1 label): {total_samples - clean_count} ({100 - clean_pct:.4f}%)\n"
    )

    # 3. Input Data Analysis (Text Modality)
    print("=== INPUT DATA ANALYSIS (TEXT) ===")

    # Calculate lengths
    # Character length
    df["char_length"] = df["comment_text"].apply(len)
    # Word length (simple whitespace split)
    df["word_count"] = df["comment_text"].apply(lambda x: len(str(x).split()))

    def print_stats(name, series):
        print(f"--- {name} Statistics ---")
        print(f"Mean:   {series.mean():.4f}")
        print(f"Std:    {series.std():.4f}")
        print(f"Min:    {series.min():.4f}")
        print(f"25%:    {series.quantile(0.25):.4f}")
        print(f"Median: {series.median():.4f}")
        print(f"75%:    {series.quantile(0.75):.4f}")
        print(f"Max:    {series.max():.4f}")
        print(f"Skew:   {series.skew():.4f}")

    print_stats("Character Length", df["char_length"])
    print()
    print_stats("Word Count", df["word_count"])

    # Vocabulary Analysis
    print("\n--- Vocabulary Analysis ---")
    # We use a simple whitespace tokenizer for EDA speed and robustness
    # Sampling if dataset is massive, but 127k is manageable
    all_tokens = pd.Series(" ".join(df["comment_text"]).split())
    vocab_size = all_tokens.nunique()
    print(
        f"Approximate Vocabulary Size (unique whitespace-separated tokens): {vocab_size}"
    )

    # 4. Feature/Signal Relationships
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Correlation between labels
    print("--- Target Label Correlations (Pearson) ---")
    # Since these are binary, this is effectively Phi coefficient
    corr_matrix = df[LABEL_COLS].corr(method="pearson")

    # Print matrix in a readable format
    header = " " * 15 + "".join([f"{col[:8]:>10}" for col in LABEL_COLS])
    print(header)
    for row_label in LABEL_COLS:
        row_str = f"{row_label:<15}"
        for col_label in LABEL_COLS:
            val = corr_matrix.loc[row_label, col_label]
            row_str += f"{val:>10.4f}"
        print(row_str)

    # Meta-feature relationships: Length vs Toxicity
    print("\n--- Meta-Feature Relationship: Text Length vs. Toxicity ---")

    # We define "Any Toxicity" as having at least one label
    df["is_toxic"] = (df["label_count"] > 0).astype(int)

    avg_len_clean = df[df["is_toxic"] == 0]["char_length"].mean()
    avg_len_toxic = df[df["is_toxic"] == 1]["char_length"].mean()

    print(f"Mean Char Length (Clean): {avg_len_clean:.4f}")
    print(f"Mean Char Length (Toxic): {avg_len_toxic:.4f}")

    # Point-Biserial Correlation between length and binary toxicity
    corr_len_toxic = df["char_length"].corr(df["is_toxic"])
    print(f"Correlation (Char Length vs. Any Toxicity): {corr_len_toxic:.4f}")

    # Check specific labels
    print("\nMean Character Length by Specific Label:")
    print(f"{'Label':<20} {'Present Mean':<15} {'Absent Mean':<15} {'Diff':<10}")
    for label in LABEL_COLS:
        mean_present = df[df[label] == 1]["char_length"].mean()
        mean_absent = df[df[label] == 0]["char_length"].mean()
        diff = mean_present - mean_absent
        print(f"{label:<20} {mean_present:<15.4f} {mean_absent:<15.4f} {diff:<10.4f}")

    print("\n=== EDA COMPLETE ===")


if __name__ == "__main__":
    main()
