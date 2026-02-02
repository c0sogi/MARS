import pandas as pd
import numpy as np
import os
import random
import sys


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_text_normalization_data():
    # ==========================================
    # 0. Setup and Data Loading
    # ==========================================
    set_seed(42)
    DATA_PATH = "./metadata/train.csv"

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load data with keep_default_na=False to preserve "null"/"nan" strings
    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
        dtype={
            "sentence_id": "int32",
            "token_id": "int32",
            "class": "category",
            "before": "object",
            "after": "object",
        },
    )

    total_rows = len(df)

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")
    print(f"Dataset: Text Normalization (Token-Level)")
    print(f"Total Samples: {total_rows}")
    print(f"Columns: {list(df.columns)}")
    print("")

    # ==========================================
    # 1. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")
    print("------------------------")

    # In this task, 'class' determines the normalization logic, and 'after' is the literal target.
    # We analyze 'class' distribution.
    class_counts = df["class"].value_counts()
    class_ratios = df["class"].value_counts(normalize=True)

    print("Class Distribution (Top 10):")
    for cls_name, count in class_counts.head(10).items():
        ratio = class_ratios[cls_name]
        print(f"  {cls_name:<15}: {count:>8} ({ratio*100:.4f}%)")

    if len(class_counts) > 10:
        print(f"  ... and {len(class_counts) - 10} more classes.")

    # Analyze the 'after' column (Target Text)
    # We check how often 'before' == 'after' (No Normalization Needed)
    df["is_changed"] = df["before"] != df["after"]
    change_count = df["is_changed"].sum()
    change_ratio = change_count / total_rows

    print(f"\nNormalization Change Rate:")
    print(
        f"  Unchanged (Copy): {total_rows - change_count:>8} ({(1-change_ratio)*100:.4f}%)"
    )
    print(f"  Changed (Norm)  : {change_count:>8} ({change_ratio*100:.4f}%)")

    print("")

    # ==========================================
    # 2. Input Data Analysis (Text Modality)
    # ==========================================
    print("INPUT DATA ANALYSIS (TEXT)")
    print("--------------------------")

    # Length Analysis (Character count of 'before' token)
    # Vectorized string length calculation
    df["len_before"] = df["before"].str.len()

    print("Token Length Statistics (Characters - 'before'):")
    print(f"  Mean  : {df['len_before'].mean():.4f}")
    print(f"  Std   : {df['len_before'].std():.4f}")
    print(f"  Min   : {df['len_before'].min():.4f}")
    print(f"  Max   : {df['len_before'].max():.4f}")
    print(f"  Median: {df['len_before'].median():.4f}")

    # Vocabulary Analysis
    unique_tokens = df["before"].nunique()
    print(f"\nVocabulary Statistics:")
    print(f"  Unique 'before' tokens: {unique_tokens}")
    print(f"  Vocabulary / Total Ratio: {unique_tokens/total_rows:.4f}")

    # Top tokens
    print("\nMost Common Tokens:")
    top_tokens = df["before"].value_counts().head(5)
    for token, count in top_tokens.items():
        # Escape newlines or tabs for printing
        safe_token = repr(token)
        print(f"  {safe_token:<15}: {count}")

    print("")

    # ==========================================
    # 3. Feature/Signal Relationships
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("----------------------------")

    # Relationship 1: Class vs Change Rate
    # Which classes require normalization most often?
    print("Change Rate by Class (Top 10 by Volume):")
    # Get top classes
    top_classes = class_counts.head(10).index

    # Group by class and calculate mean of is_changed
    # We use the observed=True (if categorical) or just standard groupby
    class_stats = (
        df[df["class"].isin(top_classes)]
        .groupby("class", observed=True)["is_changed"]
        .mean()
    )

    # Sort by the original volume order (top_classes)
    for cls_name in top_classes:
        rate = class_stats.loc[cls_name]
        print(f"  {cls_name:<15}: {rate*100:.4f}% changed")

    # Relationship 2: Class vs Token Length
    # Do certain classes have longer raw tokens?
    print("\nAverage Token Length by Class (Top 10 by Volume):")
    len_stats = (
        df[df["class"].isin(top_classes)]
        .groupby("class", observed=True)["len_before"]
        .mean()
    )

    for cls_name in top_classes:
        avg_len = len_stats.loc[cls_name]
        print(f"  {cls_name:<15}: {avg_len:.4f} chars")

    # Relationship 3: Sentence Context
    # Analyze sentence lengths (number of tokens per sentence)
    print("\nSentence Structure Analysis:")
    sentence_lengths = df.groupby("sentence_id").size()
    print(f"  Mean Tokens/Sentence: {sentence_lengths.mean():.4f}")
    print(f"  Max Tokens/Sentence : {sentence_lengths.max()}")
    print(f"  Min Tokens/Sentence : {sentence_lengths.min()}")

    # Correlation between length and change
    # Point-biserial correlation roughly equivalent to Pearson here since is_changed is 0/1
    corr = df["len_before"].corr(df["is_changed"].astype(int))
    print(f"\nCorrelation (Token Length vs. Is_Changed): {corr:.4f}")
    print(
        "  (Positive correlation implies longer tokens are more likely to be normalized/changed)"
    )


if __name__ == "__main__":
    analyze_text_normalization_data()
