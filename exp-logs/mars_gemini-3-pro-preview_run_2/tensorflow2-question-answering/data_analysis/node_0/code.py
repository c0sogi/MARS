import os
import json
import pandas as pd
import numpy as np
import random
from collections import Counter

# Constants
TRAIN_META_PATH = "./metadata/train_metadata.csv"
TRAIN_DATA_PATH = "./input/simplified-nq-train.jsonl"
SAMPLE_SIZE = 10000  # Number of samples to read for text statistics
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_targets(df):
    print("--- TARGET VARIABLE ANALYSIS ---")

    # 1. Long Answer Distribution
    # According to NQ format, candidate_index == -1 means no long answer
    df["has_long_answer"] = df["long_answer_index"] != -1
    long_counts = df["has_long_answer"].value_counts()
    long_ratios = df["has_long_answer"].value_counts(normalize=True)

    print("Long Answer Existence (Classification Balance):")
    for label, ratio in long_ratios.items():
        print(f"  {label}: {ratio:.4f} ({long_counts[label]} samples)")

    # 2. Short Answer Distribution
    short_counts = df["has_short_answer"].value_counts()
    short_ratios = df["has_short_answer"].value_counts(normalize=True)

    print("\nShort Answer Existence (Classification Balance):")
    for label, ratio in short_ratios.items():
        print(f"  {label}: {ratio:.4f} ({short_counts[label]} samples)")

    # 3. Yes/No Answer Distribution
    yn_counts = df["yes_no_answer"].value_counts()
    yn_ratios = df["yes_no_answer"].value_counts(normalize=True)

    print("\nYes/No Answer Distribution:")
    for label, ratio in yn_ratios.items():
        print(f"  {label}: {ratio:.4f} ({yn_counts[label]} samples)")


def analyze_text_modality(file_path, sample_size):
    print("\n--- INPUT DATA ANALYSIS (TEXT MODALITY) ---")
    print(
        f"Sampling first {sample_size} records from {file_path} for detailed text analysis..."
    )

    doc_word_counts = []
    doc_char_counts = []
    q_word_counts = []
    q_char_counts = []

    # For vocabulary analysis
    token_counter = Counter()

    # For relationship analysis
    meta_data = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break

                entry = json.loads(line)

                # Document Text
                doc_text = entry.get("document_text", "")
                doc_tokens = doc_text.split()  # Simple whitespace tokenization
                d_w_len = len(doc_tokens)
                d_c_len = len(doc_text)

                doc_word_counts.append(d_w_len)
                doc_char_counts.append(d_c_len)
                token_counter.update(doc_tokens)

                # Question Text
                q_text = entry.get("question_text", "")
                q_tokens = q_text.split()
                q_w_len = len(q_tokens)
                q_c_len = len(q_text)

                q_word_counts.append(q_w_len)
                q_char_counts.append(q_c_len)
                token_counter.update(q_tokens)

                # Extract targets for relationship analysis
                anns = entry.get("annotations", [])
                has_long = False
                has_short = False
                if anns:
                    ann = anns[0]
                    if ann.get("long_answer", {}).get("candidate_index", -1) != -1:
                        has_long = True
                    if ann.get("short_answers"):
                        has_short = True

                meta_data.append(
                    {
                        "doc_len_words": d_w_len,
                        "q_len_words": q_w_len,
                        "has_long_answer": int(has_long),
                        "has_short_answer": int(has_short),
                    }
                )

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    # Length Analysis
    print("\nDocument Lengths (Words):")
    print(f"  Mean: {np.mean(doc_word_counts):.4f}")
    print(f"  Std:  {np.std(doc_word_counts):.4f}")
    print(f"  Min:  {np.min(doc_word_counts):.4f}")
    print(f"  Max:  {np.max(doc_word_counts):.4f}")

    # Outliers (IQR Method) for Document Length
    q75, q25 = np.percentile(doc_word_counts, [75, 25])
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    outliers = sum(x > upper_bound for x in doc_word_counts)
    print(
        f"  Outliers (> {upper_bound:.2f} words): {outliers} ({outliers/len(doc_word_counts):.4f})"
    )

    print("\nQuestion Lengths (Words):")
    print(f"  Mean: {np.mean(q_word_counts):.4f}")
    print(f"  Std:  {np.std(q_word_counts):.4f}")
    print(f"  Min:  {np.min(q_word_counts):.4f}")
    print(f"  Max:  {np.max(q_word_counts):.4f}")

    # Vocabulary Analysis
    vocab_size = len(token_counter)
    # Hapax legomena (words appearing only once) as proxy for OOV potential
    hapax_legomena = sum(1 for count in token_counter.values() if count == 1)
    oov_potential = hapax_legomena / vocab_size if vocab_size > 0 else 0.0

    print("\nVocabulary Statistics:")
    print(f"  Unique Vocabulary Size: {vocab_size}")
    print(f"  Hapax Legomena (Rare words): {hapax_legomena}")
    print(f"  OOV Potential (Ratio of rare words): {oov_potential:.4f}")

    return pd.DataFrame(meta_data)


def analyze_relationships(df):
    print("\n--- FEATURE/SIGNAL RELATIONSHIPS ---")
    if df is None or df.empty:
        print("No data available for relationship analysis.")
        return

    # Correlation Matrix
    print("Correlation Matrix (Pearson):")
    corr = df.corr()
    print(corr.to_string(float_format="{:.4f}".format))

    # Specific Meta-Feature Relationships
    print("\nMeta-Feature Analysis:")

    # Does document length correlate with having a long answer?
    doc_len_long_corr = corr.loc["doc_len_words", "has_long_answer"]
    print(f"  Correlation (Doc Length vs Has Long Answer): {doc_len_long_corr:.4f}")

    # Does document length correlate with having a short answer?
    doc_len_short_corr = corr.loc["doc_len_words", "has_short_answer"]
    print(f"  Correlation (Doc Length vs Has Short Answer): {doc_len_short_corr:.4f}")

    # Does question length correlate with answerability?
    q_len_long_corr = corr.loc["q_len_words", "has_long_answer"]
    print(f"  Correlation (Question Length vs Has Long Answer): {q_len_long_corr:.4f}")

    # Check for Collinearity
    print("\nRedundancy Check (Correlation > 0.90):")
    collinear_pairs = []
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if abs(corr.iloc[i, j]) > 0.90:
                collinear_pairs.append((columns[i], columns[j], corr.iloc[i, j]))

    if collinear_pairs:
        for c1, c2, val in collinear_pairs:
            print(f"  High collinearity between {c1} and {c2}: {val:.4f}")
    else:
        print("  No highly collinear pairs found.")


def main():
    set_seed(RANDOM_SEED)

    # 1. Target Analysis using Metadata
    if os.path.exists(TRAIN_META_PATH):
        meta_df = pd.read_csv(TRAIN_META_PATH)
        analyze_targets(meta_df)
    else:
        print(f"Metadata file {TRAIN_META_PATH} not found. Skipping target analysis.")

    # 2. Text Analysis using Raw Data Sample
    if os.path.exists(TRAIN_DATA_PATH):
        sample_df = analyze_text_modality(TRAIN_DATA_PATH, SAMPLE_SIZE)

        # 3. Relationship Analysis
        analyze_relationships(sample_df)
    else:
        print(f"Raw data file {TRAIN_DATA_PATH} not found. Skipping text analysis.")


if __name__ == "__main__":
    main()
