import os
import sys
import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random

# Suppress warnings and progress bars
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df, target_col):
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # Distribution
    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: {target_col}")
    print(f"Number of Classes: {len(counts)}")
    print("\nClass Distribution:")
    for label, count in counts.items():
        prop = props[label]
        print(f"  Class '{label}': {count} samples ({prop * 100:.4f}%)")

    # Imbalance Check
    min_class = counts.min()
    max_class = counts.max()
    ratio = max_class / min_class
    print(f"\nClass Imbalance Ratio (Max/Min): {ratio:.4f}")
    if ratio > 1.5:
        print("  -> Note: Moderate to high class imbalance detected.")
    else:
        print("  -> Note: Classes are relatively balanced.")
    print("\n")


def analyze_text_data(df, text_col):
    print("INPUT DATA ANALYSIS (TEXT)")
    print("-" * 30)

    # Derive basic text features
    # Fill NaNs with empty string just in case, though dataset seems clean
    texts = df[text_col].fillna("").astype(str)

    # Character counts
    char_lens = texts.apply(len)
    # Word counts (simple whitespace split)
    word_lens = texts.apply(lambda x: len(x.split()))

    # 1. Length Analysis
    print("Sequence Length Statistics:")

    stats_metrics = ["mean", "std", "min", "max"]

    print("  Character Counts:")
    desc_char = char_lens.describe()
    for m in stats_metrics:
        print(f"    {m.capitalize()}: {desc_char[m]:.4f}")

    print("\n  Word Counts:")
    desc_word = word_lens.describe()
    for m in stats_metrics:
        print(f"    {m.capitalize()}: {desc_word[m]:.4f}")

    # 2. Vocabulary Analysis
    print("\nVocabulary Statistics:")
    # Create a simple vocabulary
    # Using a generator to avoid huge memory spike if dataset was massive,
    # though for 14k rows list comprehension is fine.
    all_tokens = [token for text in texts for token in text.split()]
    vocab_counter = Counter(all_tokens)
    vocab_size = len(vocab_counter)
    total_tokens = len(all_tokens)

    print(f"  Total Tokens: {total_tokens}")
    print(f"  Unique Vocabulary Size: {vocab_size}")

    # Estimate OOV potential: Proportion of words appearing only once (hapax legomena)
    singletons = sum(1 for count in vocab_counter.values() if count == 1)
    oov_potential = singletons / vocab_size if vocab_size > 0 else 0
    print(
        f"  Rare Tokens (Appearing once): {singletons} ({oov_potential * 100:.4f}% of Vocab)"
    )
    print("\n")

    return char_lens, word_lens


def analyze_relationships(df, target_col, char_lens, word_lens):
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)
    print("Unstructured (Meta-Feature) Relationships:")

    # Add meta-features to a temporary dataframe for analysis
    temp_df = df[[target_col]].copy()
    temp_df["char_len"] = char_lens
    temp_df["word_count"] = word_lens

    # Group by target and calculate mean stats
    grouped = temp_df.groupby(target_col)
    means = grouped.mean()
    stds = grouped.std()

    print(f"\n  Average Text Characteristics by Class '{target_col}':")
    print(f"  {'Class':<10} | {'Avg Char Len':<15} | {'Avg Word Count':<15}")
    print("  " + "-" * 45)

    for cls in means.index:
        c_mean = means.loc[cls, "char_len"]
        w_mean = means.loc[cls, "word_count"]
        print(f"  {cls:<10} | {c_mean:<15.4f} | {w_mean:<15.4f}")

    print("\n  Interpretation:")
    # Simple heuristic check
    max_w = means["word_count"].max()
    min_w = means["word_count"].min()
    diff_pct = (max_w - min_w) / min_w

    if diff_pct > 0.1:
        print(
            f"  -> Significant difference in sentence length between authors ({diff_pct*100:.2f}% spread)."
        )
        print("     Sentence length could be a useful feature.")
    else:
        print("  -> Sentence lengths are similar across authors.")

    print("\n")


def main():
    set_seed(42)

    # Paths
    TRAIN_PATH = "./metadata/train.csv"

    # 1. Data Integrity
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    df = pd.read_csv(TRAIN_PATH)

    # Determine Modality
    # Text datasets usually have 'text' or 'sentence' columns and string data.
    # Image datasets usually have file paths or pixel columns.
    # Tabular datasets have many numerical/categorical columns.

    cols = df.columns.tolist()

    # Heuristic for modality detection based on provided dataset description
    if "text" in cols and "author" in cols:
        modality = "text"
        target_col = "author"
        input_col = "text"
    else:
        # Fallback logic if column names differed, but for this task we know the schema
        modality = "tabular"
        target_col = cols[-1]  # assumption

    print(f"Detected Modality: {modality.upper()}")
    print(f"Training Set Shape: {df.shape}")
    print("\n")

    # 2. Target Variable Analysis
    analyze_target(df, target_col)

    # 3. Input Data Analysis & 4. Relationships
    if modality == "text":
        char_lens, word_lens = analyze_text_data(df, input_col)
        analyze_relationships(df, target_col, char_lens, word_lens)
    else:
        print(
            "Modality analysis for non-text types is not implemented for this specific dataset structure."
        )


if __name__ == "__main__":
    main()
