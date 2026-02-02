import pandas as pd
import numpy as np
import os
import random
import sys


# Set fixed seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()


def main():
    # 1. DATA LOADING & INTEGRITY
    # -------------------------------------------------------------------------
    # Load the training data from the metadata directory
    try:
        train_path = "./metadata/train.csv"
        if not os.path.exists(train_path):
            # Fallback for local testing if metadata isn't generated
            train_path = "./input/train.csv"

        df = pd.read_csv(train_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure we are working with the expected text columns
    text_cols = ["context", "question", "answer_text"]
    for col in text_cols:
        df[col] = df[col].astype(str)

    # 2. FEATURE ENGINEERING (META-FEATURES)
    # -------------------------------------------------------------------------
    # Calculate lengths for Context
    df["context_char_len"] = df["context"].apply(len)
    df["context_word_len"] = df["context"].apply(lambda x: len(x.split()))

    # Calculate lengths for Question
    df["question_char_len"] = df["question"].apply(len)
    df["question_word_len"] = df["question"].apply(lambda x: len(x.split()))

    # Calculate lengths for Answer (Target)
    df["answer_char_len"] = df["answer_text"].apply(len)
    df["answer_word_len"] = df["answer_text"].apply(lambda x: len(x.split()))

    # Calculate Relative Position of Answer
    # Avoid division by zero
    df["answer_start_ratio"] = df["answer_start"] / df["context_char_len"].replace(0, 1)

    print("==== DATASET OVERVIEW ====")
    print(f"Total Training Samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 30)

    # 3. TARGET VARIABLE ANALYSIS (Answer Text)
    # -------------------------------------------------------------------------
    print("\n==== TARGET VARIABLE ANALYSIS ====")
    # Since this is a QA extraction task, we analyze the properties of the answer span.

    # Distribution of Answer Lengths
    print("--- Answer Length Statistics (Words) ---")
    ans_mean = df["answer_word_len"].mean()
    ans_std = df["answer_word_len"].std()
    ans_min = df["answer_word_len"].min()
    ans_max = df["answer_word_len"].max()

    print(f"Mean: {ans_mean:.4f}")
    print(f"Std Dev: {ans_std:.4f}")
    print(f"Min: {ans_min:.4f}")
    print(f"Max: {ans_max:.4f}")

    # Check for empty answers
    empty_answers = (df["answer_char_len"] == 0).sum()
    print(f"Empty Answers Count: {empty_answers}")

    # Outlier Analysis for Answers (IQR Method)
    Q1 = df["answer_word_len"].quantile(0.25)
    Q3 = df["answer_word_len"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[
        (df["answer_word_len"] < lower_bound) | (df["answer_word_len"] > upper_bound)
    ]
    print(
        f"Answer Length Outliers (IQR Method): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)"
    )

    # 4. INPUT DATA ANALYSIS (Text Modality)
    # -------------------------------------------------------------------------
    print("\n==== INPUT DATA ANALYSIS ====")

    # Language Distribution
    print("--- Language Distribution ---")
    lang_counts = df["language"].value_counts()
    for lang, count in lang_counts.items():
        ratio = count / len(df)
        print(f"{lang}: {count} ({ratio:.4f})")

    # Context Analysis
    print("\n--- Context Statistics (Words) ---")
    ctx_mean = df["context_word_len"].mean()
    ctx_std = df["context_word_len"].std()
    ctx_max = df["context_word_len"].max()
    print(f"Mean Length: {ctx_mean:.4f}")
    print(f"Std Dev: {ctx_std:.4f}")
    print(f"Max Length: {ctx_max:.4f}")

    # Question Analysis
    print("\n--- Question Statistics (Words) ---")
    q_mean = df["question_word_len"].mean()
    q_std = df["question_word_len"].std()
    q_max = df["question_word_len"].max()
    print(f"Mean Length: {q_mean:.4f}")
    print(f"Std Dev: {q_std:.4f}")
    print(f"Max Length: {q_max:.4f}")

    # Vocabulary Analysis
    # We'll compute unique tokens across all text columns to estimate vocab size.
    # Simple whitespace tokenization is used here.
    print("\n--- Vocabulary Analysis ---")

    def get_vocab_size(series):
        vocab = set()
        for text in series:
            vocab.update(text.lower().split())
        return len(vocab)

    context_vocab = get_vocab_size(df["context"])
    question_vocab = get_vocab_size(df["question"])

    print(f"Context Vocabulary Size (Unique Tokens): {context_vocab}")
    print(f"Question Vocabulary Size (Unique Tokens): {question_vocab}")

    # Check overlap
    # Note: This can be memory intensive on huge datasets, but manageable for this scale (800 rows)
    all_context_tokens = set()
    for t in df["context"]:
        all_context_tokens.update(t.lower().split())

    all_question_tokens = set()
    for t in df["question"]:
        all_question_tokens.update(t.lower().split())

    overlap = len(all_context_tokens.intersection(all_question_tokens))
    print(f"Vocabulary Overlap (Context & Question): {overlap}")

    # 5. FEATURE/SIGNAL RELATIONSHIPS
    # -------------------------------------------------------------------------
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # Correlation Analysis (Meta-features)
    # We check if longer contexts lead to longer answers, or if answer position is related to length.
    numeric_cols = [
        "context_word_len",
        "question_word_len",
        "answer_word_len",
        "answer_start",
        "answer_start_ratio",
    ]

    corr_matrix = df[numeric_cols].corr(method="pearson")

    print("--- Correlation Matrix (Pearson) ---")
    # Print the correlation of features with the target properties (answer length and start)
    target_corrs = corr_matrix[["answer_word_len", "answer_start"]]
    print(target_corrs)

    print("\n--- Key Observations ---")
    ctx_ans_corr = corr_matrix.loc["context_word_len", "answer_word_len"]
    q_ans_corr = corr_matrix.loc["question_word_len", "answer_word_len"]
    print(f"Correlation (Context Length vs Answer Length): {ctx_ans_corr:.4f}")
    print(f"Correlation (Question Length vs Answer Length): {q_ans_corr:.4f}")

    # Analysis by Language
    print("\n--- Target Properties by Language ---")
    grouped = df.groupby("language")[["answer_word_len", "context_word_len"]].mean()
    print(grouped)


if __name__ == "__main__":
    main()
