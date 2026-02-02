import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, pearsonr
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import time


def main():
    # 1. Setup and Configuration
    start_time = time.time()
    warnings.filterwarnings("ignore")
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    # Set random seeds for reproducibility
    SEED = 42
    np.random.seed(SEED)

    print("=== DATA LOADING ===")
    data_path = "./metadata/train.csv"
    # Loading specific columns to optimize, though memory is sufficient.
    # We need comment_text, target, and identity columns.
    df = pd.read_csv(data_path)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Define columns
    target_col = "target"
    text_col = "comment_text"
    identity_cols = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]

    # 2. Target Variable Analysis
    print("\n=== TARGET VARIABLE ANALYSIS ===")
    targets = df[target_col]

    # Distribution stats
    mean_target = targets.mean()
    std_target = targets.std()
    min_target = targets.min()
    max_target = targets.max()

    print(f"Target Mean: {mean_target:.4f}")
    print(f"Target Std:  {std_target:.4f}")
    print(f"Target Min:  {min_target:.4f}")
    print(f"Target Max:  {max_target:.4f}")

    # Skewness and Kurtosis (Regression view)
    target_skew = skew(targets)
    target_kurt = kurtosis(targets)
    print(f"Target Skewness: {target_skew:.4f}")
    print(f"Target Kurtosis: {target_kurt:.4f}")

    # Class Balance (Classification view, threshold >= 0.5)
    binary_targets = (targets >= 0.5).astype(int)
    pos_count = binary_targets.sum()
    neg_count = len(binary_targets) - pos_count
    pos_ratio = pos_count / len(binary_targets)

    print(f"Binary Class Balance (Threshold 0.5):")
    print(f"  Positive (Toxic): {pos_count} ({pos_ratio*100:.4f}%)")
    print(f"  Negative (Non-Toxic): {neg_count} ({(1-pos_ratio)*100:.4f}%)")

    # 3. Input Data Analysis (Text Modality)
    print("\n=== INPUT DATA ANALYSIS (TEXT) ===")

    # Handle NaNs in text if any (though unlikely in this clean dataset, good practice)
    if df[text_col].isnull().any():
        print(
            f"Found {df[text_col].isnull().sum()} NaN values in text column. Filling with empty string."
        )
        df[text_col] = df[text_col].fillna("")

    # Length Analysis
    # Character counts
    char_lengths = df[text_col].str.len()
    mean_char = char_lengths.mean()
    std_char = char_lengths.std()
    max_char = char_lengths.max()

    # Word counts (simple split)
    # Using a sample for very fast estimation if dataset was huge, but 1.4M is manageable.
    # We'll do direct calculation.
    word_counts = df[text_col].str.split().str.len()
    mean_word = word_counts.mean()
    std_word = word_counts.std()
    max_word = word_counts.max()

    print(f"Sequence Lengths (Characters):")
    print(f"  Mean: {mean_char:.4f}")
    print(f"  Std:  {std_char:.4f}")
    print(f"  Max:  {max_char:.4f}")

    print(f"Sequence Lengths (Words):")
    print(f"  Mean: {mean_word:.4f}")
    print(f"  Std:  {std_word:.4f}")
    print(f"  Max:  {max_word:.4f}")

    # Vocabulary Analysis
    # We use CountVectorizer to efficiently count unique tokens.
    # We limit to a reasonable sample size to keep runtime low if full scan is too slow,
    # but with 1.4M rows and simple English, a full scan is feasible within minutes.
    print("Analyzing Vocabulary (this may take a moment)...")
    try:
        # Using a simple token pattern, removing English stop words to get a sense of 'content' vocab
        vec = CountVectorizer(stop_words="english", max_features=1000000)
        vec.fit(df[text_col])
        vocab_size = len(vec.vocabulary_)
        print(f"Unique Vocabulary Size (ignoring stop_words): {vocab_size}")
    except Exception as e:
        print(f"Vocabulary analysis skipped due to memory/time constraints: {e}")

    # 4. Feature/Signal Relationships
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Meta-Feature Relationships: Length vs Target
    # Do longer comments tend to be more toxic?
    corr_char, _ = pearsonr(char_lengths, targets)
    corr_word, _ = pearsonr(word_counts, targets)

    print(f"Correlation (Character Length vs Target): {corr_char:.4f}")
    print(f"Correlation (Word Count vs Target):       {corr_word:.4f}")

    # Identity Analysis (Bias Check)
    # We check the correlation between the presence of an identity and the toxicity target.
    # This informs the 'Unintended Bias' aspect of the task.
    print("\nIdentity vs Target Correlations (Pearson):")

    # Filter for identity columns that exist in the dataframe
    existing_ids = [col for col in identity_cols if col in df.columns]

    if existing_ids:
        correlations = {}
        for col in existing_ids:
            # Fill NaNs in identity columns with 0 (assumption: not mentioned)
            # The dataset description says these are fractional, but NaN usually implies not annotated/not present.
            # However, in this specific dataset, NaNs are common for rows not annotated for identities.
            # We calculate correlation only on rows where the identity value is not NaN to be precise,
            # or fill 0 if we assume NaN means not present.
            # Standard practice for this dataset: NaN in identity columns means "not annotated".
            # We will calculate correlation on the subset where identity is not null.

            subset = df.dropna(subset=[col])
            if len(subset) > 0:
                corr, _ = pearsonr(subset[col], subset[target_col])
                correlations[col] = corr
            else:
                correlations[col] = 0.0

        # Sort by absolute correlation
        sorted_corrs = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for name, corr in sorted_corrs:
            print(f"  {name}: {corr:.4f}")

        # Redundancy Check among Identities
        # Are some identities highly correlated with others? (e.g. Black + White mentions?)
        print("\nTop Identity Co-occurrence Correlations (> 0.10):")
        # We compute a correlation matrix for identities
        id_df = df[existing_ids].fillna(0)
        corr_matrix = id_df.corr().abs()

        # Extract upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find pairs
        high_corr_pairs = [
            (column, index, upper.loc[index, column])
            for column in upper.columns
            for index in upper.index
            if upper.loc[index, column] > 0.10
        ]

        if not high_corr_pairs:
            print("  No high correlations (> 0.10) found between identity attributes.")
        else:
            # Sort by correlation
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            for col, idx, val in high_corr_pairs:
                print(f"  {idx} - {col}: {val:.4f}")
    else:
        print("  No identity columns found for analysis.")

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
