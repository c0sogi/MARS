import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import skew, kurtosis, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# ==========================================
# Configuration & Seeding
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


def main():
    # ==========================================
    # 1. Load Data
    # ==========================================
    data_path = "./metadata/train.csv"
    # Reading only necessary columns for EDA to save memory if needed,
    # but full load is safe given the constraints (220GB RAM).
    df = pd.read_csv(data_path)

    # Identify column groups
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
    # Ensure binary target exists (as per task description)
    if "binary_target" not in df.columns:
        df["binary_target"] = (df["target"] >= 0.5).astype(int)

    print("==========================================")
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("==========================================")
    print(f"Dataset Shape: {df.shape}")

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # Continuous Target Analysis
    target_mean = df["target"].mean()
    target_std = df["target"].std()
    target_skew = skew(df["target"])
    target_kurt = kurtosis(df["target"])

    print(f"Target Type: Continuous (Fractional Toxicity)")
    print(f"Mean: {target_mean:.4f}, Std: {target_std:.4f}")
    print(f"Skewness: {target_skew:.4f}, Kurtosis: {target_kurt:.4f}")

    # Binary Classification Analysis
    pos_count = df["binary_target"].sum()
    neg_count = len(df) - pos_count
    pos_ratio = pos_count / len(df)

    print(f"\nBinary Class Balance (Threshold >= 0.5):")
    print(f"Positive (Toxic): {pos_count} ({pos_ratio:.2%})")
    print(f"Negative (Non-Toxic): {neg_count} ({1-pos_ratio:.2%})")
    print(f"Imbalance Ratio (Neg/Pos): {neg_count/pos_count:.4f}")

    # ==========================================
    # 3. Text Data Analysis
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TEXT)")
    print("-" * 30)

    # Handle missing text
    if df["comment_text"].isnull().any():
        print(
            f"Warning: Found {df['comment_text'].isnull().sum()} rows with NaN text. Filling with empty string."
        )
        df["comment_text"] = df["comment_text"].fillna("")

    # Calculate lengths
    # Using numpy vectorization for speed
    texts = df["comment_text"].astype(str).values
    char_lens = np.char.str_len(texts)

    # Word counts (simple whitespace split)
    # For very large lists, list comprehension is often faster than pandas apply
    word_lens = np.array([len(t.split()) for t in texts])

    print("Sequence Lengths (Character Count):")
    print(f"  Mean: {np.mean(char_lens):.4f}, Std: {np.std(char_lens):.4f}")
    print(f"  Min: {np.min(char_lens)}, Max: {np.max(char_lens)}")
    print(f"  95th Percentile: {np.percentile(char_lens, 95):.4f}")

    print("\nSequence Lengths (Word Count):")
    print(f"  Mean: {np.mean(word_lens):.4f}, Std: {np.std(word_lens):.4f}")
    print(f"  Min: {np.min(word_lens)}, Max: {np.max(word_lens)}")
    print(f"  95th Percentile: {np.percentile(word_lens, 95):.4f}")

    # Vocabulary Analysis (Approximation on full set)
    # Using a Counter on a stream of words to avoid massive memory spike of joining all text
    vocab_counter = Counter()
    # Processing in chunks to be safe and somewhat fast
    chunk_size = 50000
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        for text in chunk:
            vocab_counter.update(text.split())

    vocab_size = len(vocab_counter)
    print(f"\nVocabulary Size (approx unique tokens by whitespace): {vocab_size}")
    print(f"Top 10 Most Common Tokens: {vocab_counter.most_common(10)}")

    # ==========================================
    # 4. Feature/Signal Relationships (Structured)
    # ==========================================
    print("\nFEATURE RELATIONSHIPS (STRUCTURED / IDENTITY METADATA)")
    print("-" * 30)

    # Focus on Identity Columns provided in the task description
    # These columns contain NaNs where the identity was not annotated.
    # We fill NaNs with 0 for correlation analysis (assuming NaN implies not mentioned/unknown).

    print("Identity Attribute Analysis:")

    # Missing Values in Identities
    nan_counts = df[identity_cols].isnull().sum()
    print("Missing Values per Identity Column (Count):")
    for col, val in nan_counts.items():
        if val > 0:
            print(f"  {col}: {val} ({val/len(df):.2%})")

    # Impute for analysis
    df_identities = df[identity_cols].fillna(0.0)

    # Correlation with Target
    print("\nCorrelation with Toxicity Target (Pearson):")
    correlations = {}
    for col in identity_cols:
        corr, _ = pearsonr(df_identities[col], df["target"])
        correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for col, corr in sorted_corr:
        print(f"  {col}: {corr:.4f}")

    # Feature Importance (Lightweight Random Forest)
    print("\nIdentity Feature Importance (Predicting Binary Toxicity):")
    # Subsample for speed if necessary, but 1.4M is manageable for a small tree
    # Using a subset of 100k rows to ensure < 1 hour runtime guarantee
    sample_idx = np.random.choice(len(df), size=min(100000, len(df)), replace=False)
    X_sample = df_identities.iloc[sample_idx]
    y_sample = df["binary_target"].iloc[sample_idx]

    rf = RandomForestClassifier(
        n_estimators=20, max_depth=7, n_jobs=-1, random_state=42
    )
    rf.fit(X_sample, y_sample)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Identity Features for Toxicity Prediction:")
    for i in range(min(5, len(identity_cols))):
        feat_name = identity_cols[indices[i]]
        imp = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {imp:.4f}")

    # ==========================================
    # 5. Feature/Signal Relationships (Unstructured)
    # ==========================================
    print("\nFEATURE RELATIONSHIPS (UNSTRUCTURED / META-FEATURES)")
    print("-" * 30)

    # Correlation between Length and Target
    len_corr, _ = pearsonr(word_lens, df["target"])
    print(f"Correlation (Word Count vs Target): {len_corr:.4f}")

    # Compare lengths of Toxic vs Non-Toxic
    toxic_lens = word_lens[df["binary_target"] == 1]
    nontoxic_lens = word_lens[df["binary_target"] == 0]

    print(f"Mean Word Count (Toxic): {np.mean(toxic_lens):.4f}")
    print(f"Mean Word Count (Non-Toxic): {np.mean(nontoxic_lens):.4f}")

    print("\nAnalysis Complete.")


if __name__ == "__main__":
    main()
