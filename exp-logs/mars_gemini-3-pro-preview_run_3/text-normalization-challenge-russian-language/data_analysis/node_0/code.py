import pandas as pd
import numpy as np
import sys
import os
import random

# Constants
TRAIN_META_PATH = "./metadata/train.csv"
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_data():
    set_seed(RANDOM_SEED)

    # Check if file exists
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Error: {TRAIN_META_PATH} not found.")
        return

    # Load Data
    # Using specific dtypes to optimize memory and ensure correct parsing
    try:
        df = pd.read_csv(
            TRAIN_META_PATH,
            dtype={
                "sentence_id": str,
                "token_id": str,
                "before": str,
                "after": str,
                "class": str,
            },
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Handle NaNs in text columns (treat as empty strings for length calcs)
    df["before"] = df["before"].fillna("")
    df["after"] = df["after"].fillna("")

    # ---------------------------------------------------------
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("TARGET VARIABLE ANALYSIS")

    # 1. Class Distribution (The 'class' column is the primary category for normalization logic)
    if "class" in df.columns:
        class_counts = df["class"].value_counts()
        total_samples = len(df)

        print("\nClass Distribution (Top 10):")
        for cls, count in class_counts.head(10).items():
            ratio = count / total_samples
            print(f"  {cls:<15}: {count} ({ratio:.4%})")

        # Imbalance check
        print(f"\nTotal Classes: {len(class_counts)}")
        most_common_ratio = class_counts.iloc[0] / total_samples
        least_common_ratio = class_counts.iloc[-1] / total_samples
        print(f"Most Common Class Ratio: {most_common_ratio:.4f}")
        print(f"Least Common Class Ratio: {least_common_ratio:.4f}")
    else:
        print("\n'class' column not found.")

    # 2. Normalization Change Distribution (Target: 'after')
    # We define the 'target' behavior as whether the text changes or not.
    df["is_changed"] = df["before"] != df["after"]
    change_counts = df["is_changed"].value_counts()
    change_ratio = df["is_changed"].mean()

    print("\nNormalization Change Distribution:")
    print(
        f"  Unchanged (before == after): {change_counts.get(False, 0)} ({(1-change_ratio):.4%})"
    )
    print(
        f"  Changed   (before != after): {change_counts.get(True, 0)} ({change_ratio:.4%})"
    )

    # ---------------------------------------------------------
    # SECTION 2: INPUT DATA ANALYSIS (TEXT MODALITY)
    # ---------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # 1. Length Analysis
    # Calculate lengths of 'before' tokens
    df["len_before"] = df["before"].astype(str).apply(len)

    print("\nInput Token Length Statistics (Characters):")
    print(f"  Mean: {df['len_before'].mean():.4f}")
    print(f"  Std : {df['len_before'].std():.4f}")
    print(f"  Min : {df['len_before'].min()}")
    print(f"  Max : {df['len_before'].max()}")

    # Outlier detection using IQR
    Q1 = df["len_before"].quantile(0.25)
    Q3 = df["len_before"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[df["len_before"] > upper_bound]
    print(
        f"  Outlier Count (> {upper_bound} chars): {len(outliers)} ({len(outliers)/len(df):.4%})"
    )

    # 2. Vocabulary Analysis
    unique_tokens = df["before"].nunique()
    print("\nVocabulary Statistics:")
    print(f"  Unique Input Tokens: {unique_tokens}")
    print(f"  Vocabulary/Total Ratio: {unique_tokens/len(df):.4f}")

    # Check for character types (Russian/Cyrillic vs others)
    # Sampling for speed if dataset is huge, but here we can do a quick check on unique chars
    # We'll concatenate a sample of unique tokens to check character set
    sample_tokens = (
        df["before"].sample(min(10000, len(df)), random_state=RANDOM_SEED).astype(str)
    )
    all_chars = set("".join(sample_tokens))
    has_digits = any(c.isdigit() for c in all_chars)
    has_cyrillic = any("а" <= c.lower() <= "я" for c in all_chars)
    has_latin = any("a" <= c.lower() <= "z" for c in all_chars)

    print(f"  Contains Digits: {has_digits}")
    print(f"  Contains Cyrillic: {has_cyrillic}")
    print(f"  Contains Latin: {has_latin}")

    # ---------------------------------------------------------
    # SECTION 3: FEATURE/SIGNAL RELATIONSHIPS
    # ---------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # 1. Class vs Input Length
    # Do certain classes have distinct length profiles?
    print("\nMean Input Length by Class (Top 5 Classes):")
    if "class" in df.columns:
        top_classes = class_counts.head(5).index
        class_stats = (
            df[df["class"].isin(top_classes)].groupby("class")["len_before"].mean()
        )
        for cls, mean_len in class_stats.items():
            print(f"  {cls:<10}: {mean_len:.4f}")

    # 2. Class vs Change Probability
    # Which classes are most likely to require normalization?
    print("\nNormalization Probability by Class (Top 5 Classes):")
    if "class" in df.columns:
        change_stats = (
            df[df["class"].isin(top_classes)].groupby("class")["is_changed"].mean()
        )
        for cls, prob in change_stats.items():
            print(f"  {cls:<10}: {prob:.4f}")

    # 3. Input Length vs Output Length Correlation
    # Is the output length proportional to input length?
    df["len_after"] = df["after"].astype(str).apply(len)

    # Calculate correlation
    # We filter out extreme outliers for a more robust correlation, or just use the whole set.
    # Using whole set here.
    correlation = df["len_before"].corr(df["len_after"])
    print(f"\nCorrelation between Input and Output Lengths: {correlation:.4f}")

    # Check correlation for 'changed' items only (where normalization actually happens)
    changed_df = df[df["is_changed"]]
    if not changed_df.empty:
        corr_changed = changed_df["len_before"].corr(changed_df["len_after"])
        print(f"Correlation (Changed items only): {corr_changed:.4f}")
    else:
        print("Correlation (Changed items only): N/A (No changes found)")

    # 4. Meta-feature: Does length predict change?
    # Biserial correlation proxy: Compare mean length of changed vs unchanged
    mean_len_changed = df[df["is_changed"]]["len_before"].mean()
    mean_len_unchanged = df[~df["is_changed"]]["len_before"].mean()
    print("\nRelationship between Input Length and Change Event:")
    print(f"  Mean Length (Changed Tokens)  : {mean_len_changed:.4f}")
    print(f"  Mean Length (Unchanged Tokens): {mean_len_unchanged:.4f}")


if __name__ == "__main__":
    analyze_data()
