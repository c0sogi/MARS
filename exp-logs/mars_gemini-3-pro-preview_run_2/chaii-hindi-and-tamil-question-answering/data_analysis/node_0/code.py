import os
import pandas as pd
import numpy as np
import random
import warnings
from collections import Counter

# Suppress warnings and set silent modes
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def get_text_stats(series):
    """Calculates length statistics for a text series."""
    # Drop NaNs for calculation
    clean_series = series.dropna().astype(str)

    char_lens = clean_series.apply(len)
    word_lens = clean_series.apply(lambda x: len(x.split()))

    stats = {
        "char_mean": char_lens.mean(),
        "char_std": char_lens.std(),
        "char_min": char_lens.min(),
        "char_max": char_lens.max(),
        "word_mean": word_lens.mean(),
        "word_std": word_lens.std(),
        "word_min": word_lens.min(),
        "word_max": word_lens.max(),
        "skew": word_lens.skew(),
        "kurtosis": word_lens.kurt(),
    }
    return stats, char_lens, word_lens


def analyze_vocabulary(series_list):
    """Estimates vocabulary size from a list of text series."""
    vocab = set()
    total_tokens = 0
    for series in series_list:
        clean_series = series.dropna().astype(str)
        for text in clean_series:
            tokens = text.split()
            vocab.update(tokens)
            total_tokens += len(tokens)
    return len(vocab), total_tokens


def main():
    set_seed(42)

    # 1. Load Data
    # Using metadata/train.csv to ensure no leakage into validation set
    data_path = "./metadata/train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    print("SECTION 1: DATASET OVERVIEW")
    print(f"Total Samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Modality Detection
    # Presence of context, question, answer_text indicates Text/QA modality
    print("Detected Modality: Text (Question Answering)")

    # 2. Target Variable Analysis (Answer Text & Position)
    print("\nSECTION 2: TARGET VARIABLE ANALYSIS")

    # Target 1: Answer Text
    ans_stats, ans_char_lens, ans_word_lens = get_text_stats(df["answer_text"])

    print("Target: Answer Text Lengths")
    print(
        f"Mean Word Count: {ans_stats['word_mean']:.4f} (Std: {ans_stats['word_std']:.4f})"
    )
    print(
        f"Min Word Count: {ans_stats['word_min']:.4f}, Max: {ans_stats['word_max']:.4f}"
    )
    print(f"Skewness: {ans_stats['skew']:.4f}, Kurtosis: {ans_stats['kurtosis']:.4f}")

    # Target 2: Answer Start Position
    if "answer_start" in df.columns:
        start_mean = df["answer_start"].mean()
        start_std = df["answer_start"].std()
        print("\nTarget: Answer Start Position (Character Index)")
        print(f"Mean Start Position: {start_mean:.4f}")
        print(f"Std Start Position: {start_std:.4f}")

    # Language Distribution (Class Balance equivalent)
    if "language" in df.columns:
        print("\nTarget/Group: Language Distribution")
        lang_counts = df["language"].value_counts(normalize=True)
        for lang, ratio in lang_counts.items():
            print(f"{lang}: {ratio:.4f}")

    # 3. Input Data Analysis (Context & Question)
    print("\nSECTION 3: INPUT DATA ANALYSIS (TEXT)")

    # Context Analysis
    ctx_stats, ctx_char_lens, ctx_word_lens = get_text_stats(df["context"])
    print("Input: Context Lengths")
    print(
        f"Mean Char Count: {ctx_stats['char_mean']:.4f}, Mean Word Count: {ctx_stats['word_mean']:.4f}"
    )
    print(f"Max Word Count: {ctx_stats['word_max']:.4f}")

    # Question Analysis
    q_stats, q_char_lens, q_word_lens = get_text_stats(df["question"])
    print("\nInput: Question Lengths")
    print(
        f"Mean Char Count: {q_stats['char_mean']:.4f}, Mean Word Count: {q_stats['word_mean']:.4f}"
    )

    # Vocabulary Analysis
    # Combine context and question for vocab estimation
    vocab_size, total_tokens = analyze_vocabulary([df["context"], df["question"]])
    print("\nInput: Vocabulary Stats (Whitespace Tokenization)")
    print(f"Unique Tokens (Approx Vocab Size): {vocab_size}")
    print(f"Total Tokens: {total_tokens}")
    if total_tokens > 0:
        print(f"Type-Token Ratio: {(vocab_size / total_tokens):.4f}")

    # Missing Values
    print("\nInput: Missing Values")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"{col}: {count} ({count/len(df):.4%})")
        else:
            print(f"{col}: 0 (0.0000%)")

    # 4. Feature/Signal Relationships
    print("\nSECTION 4: FEATURE/SIGNAL RELATIONSHIPS")

    # Create temporary numeric features for correlation
    df_eda = df.copy()
    df_eda["context_len_char"] = ctx_char_lens
    df_eda["question_len_char"] = q_char_lens
    df_eda["answer_len_char"] = ans_char_lens

    # Correlation between lengths
    corr_ctx_ans = df_eda["context_len_char"].corr(df_eda["answer_len_char"])
    corr_q_ans = df_eda["question_len_char"].corr(df_eda["answer_len_char"])

    print("Correlations (Pearson)")
    print(f"Context Length vs Answer Length: {corr_ctx_ans:.4f}")
    print(f"Question Length vs Answer Length: {corr_q_ans:.4f}")

    # Relative Answer Position
    # Where does the answer usually appear? (Start / Middle / End)
    # Normalized position = answer_start / context_len
    if "answer_start" in df.columns:
        df_eda["rel_start"] = df_eda["answer_start"] / df_eda["context_len_char"]
        rel_mean = df_eda["rel_start"].mean()
        print(
            f"\nMean Normalized Answer Start Position (0=Start, 1=End): {rel_mean:.4f}"
        )
        print(
            "Interpretation: "
            + (
                "Answers tend to be in the first half."
                if rel_mean < 0.5
                else "Answers tend to be in the second half."
            )
        )

    # Language Specific Analysis (Meta-Feature Relationship)
    if "language" in df.columns:
        print("\nRelationship: Language vs Lengths")
        lang_groups = df_eda.groupby("language")[
            ["context_len_char", "answer_len_char"]
        ].mean()
        for lang in lang_groups.index:
            c_len = lang_groups.loc[lang, "context_len_char"]
            a_len = lang_groups.loc[lang, "answer_len_char"]
            print(
                f"Language '{lang}': Mean Context Len = {c_len:.4f}, Mean Answer Len = {a_len:.4f}"
            )


if __name__ == "__main__":
    main()
