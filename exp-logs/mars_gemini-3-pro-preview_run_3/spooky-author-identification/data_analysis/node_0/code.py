import pandas as pd
import numpy as np
import os
import random
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target_variable(df, target_col):
    """Analyzes the distribution and balance of the target variable."""
    print("=== TARGET VARIABLE ANALYSIS ===")

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset.")
        return

    # Determine if Classification or Regression based on data type and cardinality
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    num_unique = df[target_col].nunique()

    if not is_numeric or num_unique < 20:
        # Treat as Classification
        print("Type: Classification")
        counts = df[target_col].value_counts()
        total = len(df)

        print("\nDistribution (Counts):")
        print(counts)

        print("\nClass Balance Ratios:")
        for cls, count in counts.items():
            ratio = count / total
            print(f"  Class '{cls}': {ratio:.4f}")

        # Check for imbalance
        max_ratio = counts.max() / total
        min_ratio = counts.min() / total
        if max_ratio / min_ratio > 1.5:
            print(
                f"\nNote: Classes are imbalanced (Max/Min ratio: {max_ratio/min_ratio:.4f})"
            )
        else:
            print("\nNote: Classes are relatively balanced.")

    else:
        # Treat as Regression
        print("Type: Regression")
        print("\nDistribution Statistics:")
        print(f"  Mean: {df[target_col].mean():.4f}")
        print(f"  Std : {df[target_col].std():.4f}")
        print(f"  Min : {df[target_col].min():.4f}")
        print(f"  Max : {df[target_col].max():.4f}")

        skew = df[target_col].skew()
        kurt = df[target_col].kurt()
        print(f"\nNormality Check:")
        print(f"  Skewness: {skew:.4f} (Values > 1 or < -1 indicate high skew)")
        print(f"  Kurtosis: {kurt:.4f}")


def analyze_text_modality(df, text_col):
    """Performs analysis specific to text data."""
    print("\n=== INPUT DATA ANALYSIS (TEXT) ===")

    # Ensure text is string
    texts = df[text_col].fillna("").astype(str)

    # 1. Length Analysis
    char_counts = texts.apply(len)
    word_counts = texts.apply(lambda x: len(x.split()))

    print("Sequence Lengths (Character Counts):")
    print(f"  Mean: {char_counts.mean():.4f}")
    print(f"  Std : {char_counts.std():.4f}")
    print(f"  Min : {char_counts.min():.4f}")
    print(f"  Max : {char_counts.max():.4f}")

    # Outlier detection for lengths (IQR method)
    Q1 = char_counts.quantile(0.25)
    Q3 = char_counts.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (
        (char_counts < (Q1 - 1.5 * IQR)) | (char_counts > (Q3 + 1.5 * IQR))
    ).sum()
    print(f"  Outliers (IQR method): {outliers} samples")

    print("\nSequence Lengths (Word Counts):")
    print(f"  Mean: {word_counts.mean():.4f}")
    print(f"  Std : {word_counts.std():.4f}")
    print(f"  Min : {word_counts.min():.4f}")
    print(f"  Max : {word_counts.max():.4f}")

    # 2. Vocabulary Analysis
    # Using a set for efficiency
    vocab = set()
    total_tokens = 0
    for t in texts:
        tokens = t.split()
        vocab.update(tokens)
        total_tokens += len(tokens)

    vocab_size = len(vocab)
    # Type-Token Ratio (TTR): Measures lexical diversity.
    # Low TTR -> Repetitive text. High TTR -> Diverse vocabulary (Higher OOV potential).
    ttr = vocab_size / total_tokens if total_tokens > 0 else 0

    print("\nVocabulary Statistics:")
    print(f"  Unique Vocabulary Size: {vocab_size}")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Type-Token Ratio: {ttr:.4f}")
    print(f"  OOV Potential: {'High' if ttr > 0.1 else 'Low'} (Based on TTR)")

    return char_counts, word_counts


def analyze_relationships(df, target_col, char_counts, word_counts):
    """Analyzes relationships between meta-features and the target."""
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Create a temporary dataframe for analysis
    meta_df = pd.DataFrame(
        {"target": df[target_col], "char_len": char_counts, "word_len": word_counts}
    )

    if meta_df["target"].nunique() < 20:
        # Classification: Compare means across classes
        print("Unstructured Relationships (Text Length vs Author):")

        print("\nAverage Character Length by Class:")
        char_means = meta_df.groupby("target")["char_len"].mean()
        for cls, val in char_means.items():
            print(f"  {cls}: {val:.4f}")

        print("\nAverage Word Length by Class:")
        word_means = meta_df.groupby("target")["word_len"].mean()
        for cls, val in word_means.items():
            print(f"  {cls}: {val:.4f}")

        print("\nWord Length Standard Deviation by Class:")
        word_stds = meta_df.groupby("target")["word_len"].std()
        for cls, val in word_stds.items():
            print(f"  {cls}: {val:.4f}")

    else:
        # Regression: Correlation
        print("Unstructured Relationships (Correlation with Target):")
        corr_char = meta_df["char_len"].corr(meta_df["target"])
        corr_word = meta_df["word_len"].corr(meta_df["target"])

        print(f"  Correlation (Char Length vs Target): {corr_char:.4f}")
        print(f"  Correlation (Word Length vs Target): {corr_word:.4f}")


def main():
    # 1. Setup
    set_seed(42)
    INPUT_FILE = "./metadata/train.csv"

    # 2. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # Define columns based on dataset knowledge
    target_col = "author"
    text_col = "text"

    # 3. Target Analysis
    analyze_target_variable(df, target_col)

    # 4. Modality-Specific Analysis
    # Check if text column exists to confirm modality
    if text_col in df.columns:
        char_counts, word_counts = analyze_text_modality(df, text_col)

        # 5. Feature Relationships
        analyze_relationships(df, target_col, char_counts, word_counts)
    else:
        print("Text column not found. Skipping text-specific analysis.")
        # Fallback for tabular numerical analysis could go here if needed
        pass


if __name__ == "__main__":
    main()
