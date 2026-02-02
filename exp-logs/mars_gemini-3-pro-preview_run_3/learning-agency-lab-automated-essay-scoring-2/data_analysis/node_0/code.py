import os
import pandas as pd
import numpy as np
import re
import random
from scipy.stats import skew, kurtosis

# Set constants and seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def perform_eda():
    print("=== DATA LOADING ===")
    data_path = "./metadata/train_metadata.csv"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print(f"Dataset Loaded. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Identify Target and Modality
    target_col = "score"
    text_col = "full_text"

    # ---------------------------------------------------------
    # 2. Target Variable Analysis
    # ---------------------------------------------------------
    print("\n=== TARGET VARIABLE ANALYSIS ===")
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
    else:
        # Distribution
        print("--- Distribution ---")
        counts = df[target_col].value_counts().sort_index()
        ratios = df[target_col].value_counts(normalize=True).sort_index()

        for label, count in counts.items():
            ratio = ratios[label]
            print(f"Class {label}: {count} ({ratio:.4f})")

        # Skewness and Kurtosis (treating score as numerical for regression context)
        target_vals = df[target_col].values
        t_skew = skew(target_vals)
        t_kurt = kurtosis(target_vals)

        print("\n--- Statistics ---")
        print(f"Skewness: {t_skew:.4f}")
        print(f"Kurtosis: {t_kurt:.4f}")

        # Imbalance check
        max_ratio = ratios.max()
        min_ratio = ratios.min()
        print(f"Max/Min Class Ratio: {max_ratio/min_ratio:.4f}")

    # ---------------------------------------------------------
    # 3. Input Data Analysis (Text Modality)
    # ---------------------------------------------------------
    print("\n=== INPUT DATA ANALYSIS (TEXT) ===")

    if text_col in df.columns:
        # Pre-calculation for analysis
        # Using simple whitespace splitting for speed and robustness in this environment
        # A more complex tokenizer could be used if NLTK data was guaranteed
        df["char_count"] = df[text_col].astype(str).apply(len)
        df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

        # Length Analysis
        print("--- Sequence Lengths ---")

        def print_stats(name, series):
            print(f"{name}:")
            print(f"  Mean: {series.mean():.4f}")
            print(f"  Std : {series.std():.4f}")
            print(f"  Min : {series.min():.4f}")
            print(f"  Max : {series.max():.4f}")
            print(f"  25% : {series.quantile(0.25):.4f}")
            print(f"  75% : {series.quantile(0.75):.4f}")

        print_stats("Character Counts", df["char_count"])
        print_stats("Word Counts", df["word_count"])

        # Vocabulary Analysis
        print("\n--- Vocabulary ---")
        # Simple tokenization: lowercase and split by non-alphanumeric
        # This gives a rough estimate of vocabulary size
        all_text = " ".join(df[text_col].astype(str).tolist()).lower()
        # Using regex to extract words
        tokens = re.findall(r"\b\w+\b", all_text)
        unique_tokens = set(tokens)

        print(f"Total Tokens: {len(tokens)}")
        print(f"Unique Vocabulary Size: {len(unique_tokens)}")
        print(f"Lexical Diversity (Unique/Total): {len(unique_tokens)/len(tokens):.4f}")

    # ---------------------------------------------------------
    # 4. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    if text_col in df.columns and target_col in df.columns:
        print("--- Meta-Feature Correlations ---")
        # Correlation between lengths and score
        corr_char = df["char_count"].corr(df[target_col], method="pearson")
        corr_word = df["word_count"].corr(df[target_col], method="pearson")

        print(f"Correlation (Char Count vs Score): {corr_char:.4f}")
        print(f"Correlation (Word Count vs Score): {corr_word:.4f}")

        print("\n--- Average Length by Class ---")
        # Group by score to see if longer essays score higher
        avg_len_by_score = df.groupby(target_col)[["char_count", "word_count"]].mean()
        print(avg_len_by_score.applymap(lambda x: f"{x:.4f}"))

        print("\n--- Meta-Feature Analysis ---")
        if abs(corr_word) > 0.5:
            print(
                "Observation: Strong correlation detected between essay length and score."
            )
        elif abs(corr_word) > 0.3:
            print(
                "Observation: Moderate correlation detected between essay length and score."
            )
        else:
            print(
                "Observation: Weak correlation detected between essay length and score."
            )


if __name__ == "__main__":
    set_seed(SEED)
    perform_eda()
