import pandas as pd
import numpy as np
import os
import sys
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_eda():
    set_seed(42)

    DATA_PATH = "./metadata/train.csv"

    # 1. DATA INTEGRITY & LOADING
    # ---------------------------
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load data. keep_default_na=False prevents pandas from parsing "NaN", "null" etc as missing values
    # which is important for text data where these might be actual tokens.
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

    # Convert numeric IDs back to int for proper sorting/grouping if needed,
    # though mostly we treat them as grouping keys.
    # We'll keep them as is or convert if strictly necessary.
    # Ensure text columns are strings
    df["before"] = df["before"].astype(str)
    df["after"] = df["after"].astype(str)
    df["class"] = df["class"].astype(str)

    print("DATA INTEGRITY")
    print("-" * 20)
    print(f"Source: {DATA_PATH}")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample Sentence IDs (Head): {df['sentence_id'].head(3).tolist()}")
    print("Data loaded strictly from training metadata to prevent leakage.")
    print("")

    # 2. TARGET VARIABLE ANALYSIS
    # ---------------------------
    # In this task, the 'target' is technically the 'after' text, but the 'class'
    # is the primary structural target that determines the normalization logic.
    # We also analyze the 'Change' (whether normalization is needed).

    print("TARGET VARIABLE ANALYSIS")
    print("-" * 20)

    # Class Distribution
    class_counts = df["class"].value_counts()
    total_count = len(df)

    print("1. Class Distribution (Top 10):")
    for cls, count in class_counts.head(10).items():
        ratio = count / total_count
        print(f"   {cls:<15} : {count:>8} ({ratio:.4%})")

    # Imbalance Check
    top_class = class_counts.index[0]
    top_ratio = class_counts.iloc[0] / total_count
    min_class = class_counts.index[-1]
    min_ratio = class_counts.iloc[-1] / total_count

    print(f"\n2. Class Imbalance:")
    print(f"   Dominant Class: {top_class} ({top_ratio:.4%})")
    print(f"   Rarest Class:   {min_class} ({min_ratio:.4%})")

    # Change Analysis (Is normalization required?)
    # A crucial aspect of this task is knowing when to copy vs when to transform.
    df["is_changed"] = df["before"] != df["after"]
    change_counts = df["is_changed"].value_counts()
    change_ratio = df["is_changed"].mean()

    print(f"\n3. Normalization Requirement (Target Change):")
    print(f"   Tokens requiring change: {df['is_changed'].sum()} ({change_ratio:.4%})")
    print(
        f"   Tokens unchanged (copy): {(~df['is_changed']).sum()} ({1-change_ratio:.4%})"
    )

    print("")

    # 3. INPUT DATA ANALYSIS (TEXT MODALITY)
    # --------------------------------------
    print("INPUT DATA ANALYSIS (TEXT)")
    print("-" * 20)

    # Calculate lengths
    df["len_before"] = df["before"].apply(len)
    df["len_after"] = df["after"].apply(len)

    # Length Statistics
    print("1. Token Length Statistics (Characters):")
    stats = df["len_before"].describe()
    print(f"   Mean Length: {stats['mean']:.4f}")
    print(f"   Std Dev:     {stats['std']:.4f}")
    print(f"   Min:         {stats['min']:.4f}")
    print(f"   Max:         {stats['max']:.4f}")

    # Outliers (IQR Method)
    Q1 = stats["25%"]
    Q3 = stats["75%"]
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[df["len_before"] > upper_bound]
    print(
        f"   Outlier Count (> {upper_bound:.2f} chars): {len(outliers)} ({len(outliers)/len(df):.4%})"
    )

    # Vocabulary Analysis
    vocab_before = df["before"].nunique()
    vocab_after = df["after"].nunique()

    print(f"\n2. Vocabulary Size (Unique Tokens):")
    print(f"   Input Vocabulary:  {vocab_before}")
    print(f"   Target Vocabulary: {vocab_after}")
    print(f"   Vocabulary Ratio (After/Before): {vocab_after/vocab_before:.4f}")

    # Character Composition
    # Check for digits in input (strong signal for normalization)
    has_digit = df["before"].str.contains(r"\d", regex=True).mean()

    # Check for non-ascii
    # A simple way to check non-ascii is encoding
    def is_ascii(s):
        try:
            s.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    # Sampling for ASCII check to save time if dataset is huge,
    # but 7M is doable. Let's do a quick check on unique tokens to speed up.
    unique_tokens = pd.Series(df["before"].unique())
    non_ascii_ratio_vocab = (~unique_tokens.apply(is_ascii)).mean()

    print(f"\n3. Character Composition:")
    print(f"   Tokens containing digits: {has_digit:.4%}")
    print(f"   Non-ASCII tokens in vocab: {non_ascii_ratio_vocab:.4%}")

    # Sentence Level Analysis
    # Group by sentence_id to see sentence lengths
    sent_counts = df.groupby("sentence_id").size()
    print(f"\n4. Sentence Level Statistics (Tokens per Sentence):")
    print(f"   Mean Tokens/Sentence: {sent_counts.mean():.4f}")
    print(f"   Max Tokens/Sentence:  {sent_counts.max():.4f}")
    print("")

    # 4. FEATURE/SIGNAL RELATIONSHIPS
    # -------------------------------
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 20)

    # Relationship 1: Class vs Change Probability
    # Which classes are most likely to change?
    print("1. Change Probability by Class (Top 10 most frequent classes):")
    top_classes = class_counts.head(10).index

    # Calculate change rate per class
    class_change_stats = (
        df[df["class"].isin(top_classes)]
        .groupby("class")["is_changed"]
        .mean()
        .sort_values(ascending=False)
    )

    for cls, rate in class_change_stats.items():
        print(f"   {cls:<15}: {rate:.4%} change rate")

    # Relationship 2: Input Length vs Class
    # Do certain classes have distinct length profiles?
    print(f"\n2. Average Input Length by Class (Top 5 most frequent):")
    top_5_classes = class_counts.head(5).index
    len_by_class = (
        df[df["class"].isin(top_5_classes)]
        .groupby("class")["len_before"]
        .mean()
        .sort_values(ascending=False)
    )

    for cls, avg_len in len_by_class.items():
        print(f"   {cls:<15}: {avg_len:.4f} chars")

    # Relationship 3: Expansion Ratio
    # How much does the text expand when normalized?
    # We only look at rows where change occurred to avoid skew from 1:1 copies.
    changed_df = df[df["is_changed"]].copy()
    if len(changed_df) > 0:
        changed_df["expansion"] = changed_df["len_after"] / changed_df[
            "len_before"
        ].replace(
            0, 1
        )  # avoid div by zero
        avg_expansion = changed_df["expansion"].mean()

        print(f"\n3. Text Expansion on Normalization:")
        print(f"   Avg Expansion Ratio (len_after / len_before): {avg_expansion:.4f}")
        print(f"   (Calculated only on tokens that changed)")

        # Check specific class expansion
        if "DATE" in changed_df["class"].unique():
            date_exp = changed_df[changed_df["class"] == "DATE"]["expansion"].mean()
            print(f"   DATE Expansion Ratio: {date_exp:.4f}")
        if "CARDINAL" in changed_df["class"].unique():
            card_exp = changed_df[changed_df["class"] == "CARDINAL"]["expansion"].mean()
            print(f"   CARDINAL Expansion Ratio: {card_exp:.4f}")
    else:
        print("\n3. Text Expansion: No changed tokens found.")


if __name__ == "__main__":
    perform_eda()
