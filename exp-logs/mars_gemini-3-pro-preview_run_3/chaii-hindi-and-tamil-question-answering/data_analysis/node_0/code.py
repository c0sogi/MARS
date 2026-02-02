import pandas as pd
import numpy as np
import random
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_text_stats(series):
    """Calculates basic statistics for a numerical series."""
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "median": series.median(),
    }


def analyze_vocabulary(text_series):
    """Estimates vocabulary size and OOV potential."""
    all_tokens = []
    for text in text_series:
        if isinstance(text, str):
            all_tokens.extend(text.lower().split())
    unique_tokens = set(all_tokens)
    return len(unique_tokens), len(all_tokens)


def main():
    # 1. Setup
    set_seed(42)
    data_path = "./metadata/train.csv"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load Data
    df = pd.read_csv(data_path)

    # Ensure text columns are strings
    text_cols = ["context", "question", "answer_text"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    # Feature Engineering for Analysis
    # Lengths in characters
    df["context_char_len"] = df["context"].apply(len)
    df["question_char_len"] = df["question"].apply(len)
    df["answer_char_len"] = df["answer_text"].apply(len)

    # Lengths in words (whitespace split)
    df["context_word_len"] = df["context"].apply(lambda x: len(x.split()))
    df["question_word_len"] = df["question"].apply(lambda x: len(x.split()))
    df["answer_word_len"] = df["answer_text"].apply(lambda x: len(x.split()))

    # Relative answer position
    # Avoid division by zero
    df["answer_rel_pos"] = df.apply(
        lambda row: (
            row["answer_start"] / row["context_char_len"]
            if row["context_char_len"] > 0
            else 0
        ),
        axis=1,
    )

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # 2. Target Variable Analysis (Answer Text)
    # In QA, the 'target' is the answer text. We analyze its length distribution.
    print("\nTARGET VARIABLE ANALYSIS")
    print("------------------------")

    ans_char_stats = get_text_stats(df["answer_char_len"])
    ans_word_stats = get_text_stats(df["answer_word_len"])

    print("Answer Length Distribution (Characters):")
    print(f"  Mean: {ans_char_stats['mean']:.4f}")
    print(f"  Std:  {ans_char_stats['std']:.4f}")
    print(f"  Min:  {ans_char_stats['min']:.4f}")
    print(f"  Max:  {ans_char_stats['max']:.4f}")

    print("\nAnswer Length Distribution (Words):")
    print(f"  Mean: {ans_word_stats['mean']:.4f}")
    print(f"  Std:  {ans_word_stats['std']:.4f}")
    print(f"  Min:  {ans_word_stats['min']:.4f}")
    print(f"  Max:  {ans_word_stats['max']:.4f}")

    # Check for empty answers
    empty_answers = (df["answer_char_len"] == 0).sum()
    print(f"\nEmpty Answers Count: {empty_answers}")

    # 3. Input Data Analysis (Text Modality)
    print("\nINPUT DATA ANALYSIS (TEXT)")
    print("--------------------------")

    # Context Analysis
    ctx_char_stats = get_text_stats(df["context_char_len"])
    ctx_word_stats = get_text_stats(df["context_word_len"])

    print("Context Lengths (Words):")
    print(f"  Mean: {ctx_word_stats['mean']:.4f}")
    print(f"  Std:  {ctx_word_stats['std']:.4f}")
    print(f"  Max:  {ctx_word_stats['max']:.4f}")

    # Question Analysis
    q_word_stats = get_text_stats(df["question_word_len"])
    print("\nQuestion Lengths (Words):")
    print(f"  Mean: {q_word_stats['mean']:.4f}")
    print(f"  Std:  {q_word_stats['std']:.4f}")
    print(f"  Max:  {q_word_stats['max']:.4f}")

    # Vocabulary Analysis
    # Combine context and question for overall vocab
    combined_text = pd.concat([df["context"], df["question"]])
    vocab_size, total_tokens = analyze_vocabulary(combined_text)

    print("\nVocabulary Statistics:")
    print(f"  Unique Vocabulary Size: {vocab_size}")
    print(f"  Total Tokens: {total_tokens}")
    print(
        f"  Type-Token Ratio: {(vocab_size/total_tokens if total_tokens > 0 else 0):.4f}"
    )

    # Language Distribution
    print("\nLanguage Distribution:")
    lang_counts = df["language"].value_counts(normalize=True)
    for lang, freq in lang_counts.items():
        print(f"  {lang}: {freq:.4f}")

    # 4. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("----------------------------")

    # Correlations
    # We look at correlations between input lengths and target lengths
    corr_matrix = df[
        ["context_word_len", "question_word_len", "answer_word_len", "answer_start"]
    ].corr()

    print("Correlations (Pearson):")
    print(
        f"  Context Length vs Answer Length: {corr_matrix.loc['context_word_len', 'answer_word_len']:.4f}"
    )
    print(
        f"  Question Length vs Answer Length: {corr_matrix.loc['question_word_len', 'answer_word_len']:.4f}"
    )
    print(
        f"  Context Length vs Answer Start Pos: {corr_matrix.loc['context_word_len', 'answer_start']:.4f}"
    )

    # Meta-Feature Relationship: Language vs Answer Length
    print("\nRelationship: Language vs Answer Length (Words)")
    lang_groups = df.groupby("language")["answer_word_len"].mean()
    for lang, mean_len in lang_groups.items():
        print(f"  Average Answer Length ({lang}): {mean_len:.4f}")

    # Meta-Feature Relationship: Answer Position Bias
    # Check if answers tend to appear at the beginning, middle, or end of the context
    print("\nRelationship: Answer Position in Context (0.0=Start, 1.0=End)")
    pos_stats = get_text_stats(df["answer_rel_pos"])
    print(f"  Mean Relative Position: {pos_stats['mean']:.4f}")
    print(f"  Median Relative Position: {pos_stats['median']:.4f}")

    # Check for "Short Context" bias
    # Do shorter contexts have longer answers relative to their length?
    df["answer_ratio"] = df["answer_word_len"] / df["context_word_len"]
    short_ctx_ratio = df[df["context_word_len"] < df["context_word_len"].median()][
        "answer_ratio"
    ].mean()
    long_ctx_ratio = df[df["context_word_len"] >= df["context_word_len"].median()][
        "answer_ratio"
    ].mean()

    print("\nRelationship: Context Length vs Answer Coverage Ratio")
    print(f"  Avg Answer/Context Ratio (Short Contexts): {short_ctx_ratio:.4f}")
    print(f"  Avg Answer/Context Ratio (Long Contexts):  {long_ctx_ratio:.4f}")


if __name__ == "__main__":
    main()
