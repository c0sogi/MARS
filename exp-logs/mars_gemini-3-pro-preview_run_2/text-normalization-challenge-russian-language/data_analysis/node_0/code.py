import pandas as pd
import numpy as np
import os
import sys
import random
import unicodedata


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_text_data():
    # Configuration
    METADATA_PATH = "./metadata/train.csv"
    SEED = 42

    set_seed(SEED)

    print("Loading training data...")
    # Load data
    try:
        df = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print(f"Error: {METADATA_PATH} not found.")
        return

    # Handle potential missing values in text columns (e.g. if token is literal "nan")
    df["before"] = df["before"].fillna("").astype(str)
    df["after"] = df["after"].fillna("").astype(str)
    df["class"] = df["class"].fillna("UNKNOWN").astype(str)

    print("Data loaded successfully.")
    print("-" * 30)

    # ==========================================
    # 1. DATA INTEGRITY
    # ==========================================
    print("DATA INTEGRITY")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique Sentences: {df['sentence_id'].nunique()}")
    print("-" * 30)

    # ==========================================
    # 2. TARGET VARIABLE ANALYSIS
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    # A. Class Distribution (Categorical Target)
    class_counts = df["class"].value_counts()
    class_ratios = df["class"].value_counts(normalize=True)

    print("Class Distribution (Top 10):")
    for cls_name, count in class_counts.head(10).items():
        ratio = class_ratios[cls_name]
        print(f"  {cls_name:<15}: {count} ({ratio:.4%})")

    if len(class_counts) > 10:
        print(f"  ... and {len(class_counts) - 10} more classes.")

    # B. Normalization Change Analysis (Derived Target Property)
    # We define the 'target' behavior as whether the token changes from before to after
    df["is_changed"] = df["before"] != df["after"]
    change_count = df["is_changed"].sum()
    change_ratio = change_count / len(df)

    print(f"\nNormalization Change Rate:")
    print(f"  Changed Tokens : {change_count} ({change_ratio:.4%})")
    print(f"  Unchanged      : {len(df) - change_count} ({1 - change_ratio:.4%})")

    # ==========================================
    # 3. INPUT DATA ANALYSIS (TEXT MODALITY)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # A. Length Analysis (Characters)
    df["len_before"] = df["before"].apply(len)
    df["len_after"] = df["after"].apply(len)

    stats_before = df["len_before"].describe()

    print("Input Token Lengths (Characters):")
    print(f"  Mean: {stats_before['mean']:.4f}")
    print(f"  Std : {stats_before['std']:.4f}")
    print(f"  Min : {stats_before['min']:.4f}")
    print(f"  Max : {stats_before['max']:.4f}")

    # Outlier Analysis (IQR Method) for Input Length
    Q1 = stats_before["25%"]
    Q3 = stats_before["75%"]
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[df["len_before"] > upper_bound]
    print(
        f"  Outliers (> {upper_bound:.2f} chars): {len(outliers)} ({len(outliers)/len(df):.4%})"
    )

    # B. Vocabulary Analysis
    vocab_size = df["before"].nunique()
    print(f"\nVocabulary Statistics:")
    print(f"  Unique Input Tokens: {vocab_size}")
    print(f"  Vocabulary/Token Ratio: {vocab_size/len(df):.4f}")

    # C. Character Set Analysis
    # Check for presence of digits, latin chars, cyrillic chars
    # We use a sample for speed if dataset is massive, but 7M is doable with vectorized string ops
    # Using regex for speed
    import re

    print("\nCharacter Composition (Input Tokens):")
    has_digit = df["before"].str.contains(r"\d", regex=True).mean()
    has_latin = df["before"].str.contains(r"[a-zA-Z]", regex=True).mean()
    has_cyrillic = df["before"].str.contains(r"[а-яА-ЯёЁ]", regex=True).mean()

    print(f"  Contains Digits   : {has_digit:.4%}")
    print(f"  Contains Latin    : {has_latin:.4%}")
    print(f"  Contains Cyrillic : {has_cyrillic:.4%}")

    # ==========================================
    # 4. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # A. Class vs Change Rate
    # Which classes require normalization most often?
    print("Change Rate by Class (Top 5 Classes by Volume):")
    top_classes = class_counts.head(5).index
    for cls in top_classes:
        subset = df[df["class"] == cls]
        subset_change_rate = subset["is_changed"].mean()
        print(f"  {cls:<10}: {subset_change_rate:.4%}")

    # B. Input Length vs Class
    # Do certain classes have longer input tokens?
    print("\nMean Input Length by Class (Top 5 Classes):")
    for cls in top_classes:
        subset = df[df["class"] == cls]
        mean_len = subset["len_before"].mean()
        print(f"  {cls:<10}: {mean_len:.4f} chars")

    # C. Input vs Output Length Correlation
    # Pearson correlation between length of 'before' and length of 'after'
    len_corr = df["len_before"].corr(df["len_after"])
    print(f"\nCorrelation (Input Length vs Output Length):")
    print(f"  Pearson r: {len_corr:.4f}")

    # D. Expansion Ratio
    # How much does the text expand/contract?
    # Avoid division by zero
    mask = df["len_before"] > 0
    expansion_ratio = (df.loc[mask, "len_after"] / df.loc[mask, "len_before"]).mean()
    print(f"  Mean Expansion Ratio (After/Before): {expansion_ratio:.4f}")

    print("-" * 30)
    print("EDA Completed.")


if __name__ == "__main__":
    analyze_text_data()
