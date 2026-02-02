import os
import random
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.feature_extraction.text import CountVectorizer


# Set fixed random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_eda():
    set_seed()

    # Define path to metadata training set
    TRAIN_PATH = "./metadata/train.csv"

    # Load Data
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    df = pd.read_csv(TRAIN_PATH)

    # === TARGET VARIABLE ANALYSIS ===
    print("=== TARGET VARIABLE ANALYSIS ===")
    target_col = "score"

    if target_col in df.columns:
        # Distribution / Class Balance
        class_counts = df[target_col].value_counts().sort_index()
        total_count = len(df)

        print("Distribution:")
        for score_val, count in class_counts.items():
            ratio = count / total_count
            print(f"  Class {score_val}: {count} ({ratio:.4f})")

        # Skewness and Kurtosis
        # Treating score as numerical to assess normality for regression approaches
        scores = df[target_col].values
        skew_val = skew(scores)
        kurt_val = kurtosis(scores)

        print(f"Skewness: {skew_val:.4f}")
        print(f"Kurtosis: {kurt_val:.4f}")
    else:
        print(f"Target column '{target_col}' not found in dataset.")
    print()

    # === INPUT DATA ANALYSIS (TEXT) ===
    print("=== INPUT DATA ANALYSIS (TEXT) ===")
    text_col = "full_text"

    if text_col in df.columns:
        # Handle potential missing values (though unlikely in this dataset)
        if df[text_col].isnull().any():
            nan_count = df[text_col].isnull().sum()
            print(f"Missing Values: {nan_count} ({nan_count/len(df):.4f})")
            df[text_col] = df[text_col].fillna("")

        # 1. Lengths Analysis
        # Character count
        df["char_count"] = df[text_col].astype(str).apply(len)
        # Word count (using simple split for efficiency and robustness)
        df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

        print("Sequence Lengths (Character Count):")
        desc_char = df["char_count"].describe()
        print(f"  Mean: {desc_char['mean']:.4f}")
        print(f"  Std:  {desc_char['std']:.4f}")
        print(f"  Min:  {desc_char['min']:.4f}")
        print(f"  Max:  {desc_char['max']:.4f}")

        print("Sequence Lengths (Word Count):")
        desc_word = df["word_count"].describe()
        print(f"  Mean: {desc_word['mean']:.4f}")
        print(f"  Std:  {desc_word['std']:.4f}")
        print(f"  Min:  {desc_word['min']:.4f}")
        print(f"  Max:  {desc_word['max']:.4f}")

        # 2. Vocabulary Analysis
        print("Vocabulary Statistics:")
        # Use CountVectorizer to build vocabulary
        # token_pattern matches words with 1 or more alphanumeric characters
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        X = vectorizer.fit_transform(df[text_col].astype(str))

        vocab_size = len(vectorizer.vocabulary_)
        print(f"  Unique Vocabulary Size: {vocab_size}")

        # Analyze OOV Potential / Rare Words
        # Calculate words that appear only once in the entire training corpus (Hapax Legomena)
        # Summing over axis 0 gives total frequency of each word
        word_freqs = np.array(X.sum(axis=0)).flatten()
        rare_words_count = np.sum(word_freqs == 1)
        rare_ratio = rare_words_count / vocab_size if vocab_size > 0 else 0

        print(f"  Rare Words (Freq=1): {rare_words_count} ({rare_ratio:.4f} of vocab)")

    else:
        print(f"Text column '{text_col}' not found in dataset.")
    print()

    # === FEATURE/SIGNAL RELATIONSHIPS ===
    print("=== FEATURE/SIGNAL RELATIONSHIPS ===")

    if target_col in df.columns and text_col in df.columns:
        print("Unstructured (Meta-Feature) Relationships:")

        # Correlation Analysis
        # Pearson correlation between lengths and score
        corr_char = df["char_count"].corr(df[target_col], method="pearson")
        corr_word = df["word_count"].corr(df[target_col], method="pearson")

        print(f"  Correlation (Char Count vs Score): {corr_char:.4f}")
        print(f"  Correlation (Word Count vs Score): {corr_word:.4f}")

        # Relationship: Do longer essays correlate with specific classes?
        print("  Mean Word Count per Score Class:")
        mean_wc_by_score = df.groupby(target_col)["word_count"].mean()
        for score_val, mean_wc in mean_wc_by_score.items():
            print(f"    Class {score_val}: {mean_wc:.4f}")

    print()


if __name__ == "__main__":
    perform_eda()
