import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)

    # Define paths
    TRAIN_PATH = "./metadata/train.parquet"

    # 1. Load Data
    # We strictly use the training set for EDA
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    df = pd.read_parquet(TRAIN_PATH)

    # Ensure string types for text columns to avoid errors with mixed types
    df["before"] = df["before"].astype(str)
    df["after"] = df["after"].astype(str)
    df["class"] = df["class"].astype(str)

    print("SECTION 1: TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # 1.1 Class Distribution
    class_counts = df["class"].value_counts()
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Unique Classes: {len(class_counts)}")
    print("\nClass Distribution (Top 10):")
    print(f"{'Class':<15} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 45)
    for cls, count in class_counts.head(10).items():
        print(f"{cls:<15} | {count:<10} | {count/total_samples*100:.4f}%")

    # 1.2 Target Text Analysis
    unique_after = df["after"].nunique()
    print(f"\nUnique Target Tokens ('after'): {unique_after}")
    print(f"Target Vocabulary Size Ratio: {unique_after/total_samples:.4f}")

    top_after = df["after"].value_counts().head(5)
    print("\nTop 5 Most Frequent Target Tokens:")
    for token, count in top_after.items():
        # Escape newlines for printing
        safe_token = token.replace("\n", "\\n")
        print(f"'{safe_token}': {count} ({count/total_samples*100:.4f}%)")

    print("\nSECTION 2: INPUT DATA ANALYSIS (TEXT MODALITY)")
    print("-" * 30)

    # 2.1 Length Analysis
    # Calculate character lengths
    df["len_before"] = df["before"].apply(len)
    df["len_after"] = df["after"].apply(len)

    print("Input Token Length ('before') Statistics:")
    print(f"Mean: {df['len_before'].mean():.4f}")
    print(f"Std:  {df['len_before'].std():.4f}")
    print(f"Min:  {df['len_before'].min()}")
    print(f"Max:  {df['len_before'].max()}")

    # 2.2 Vocabulary Analysis
    unique_before = df["before"].nunique()
    print(f"\nUnique Input Tokens ('before'): {unique_before}")
    print(f"Input Vocabulary Size Ratio: {unique_before/total_samples:.4f}")

    # 2.3 OOV Potential (Simple heuristic: tokens appearing only once)
    # Calculating exact OOV requires a test set, but we can check for 'rare' tokens in train
    # Counting tokens that appear only once
    token_counts = df["before"].value_counts()
    rare_tokens = (token_counts == 1).sum()
    print(
        f"Rare Tokens (Frequency = 1): {rare_tokens} ({rare_tokens/unique_before*100:.4f}% of vocab)"
    )

    print("\nSECTION 3: FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # 3.1 Identity Mapping (Unstructured Relationship)
    # How often is before == after? This is critical for normalization.
    df["is_unchanged"] = df["before"] == df["after"]
    unchanged_count = df["is_unchanged"].sum()
    print(
        f"Identity Mapping (before == after): {unchanged_count} ({unchanged_count/total_samples*100:.4f}%)"
    )

    print("\nIdentity Mapping by Class (Top 5 Classes):")
    # Calculate % unchanged per class
    class_unchanged = (
        df.groupby("class")["is_unchanged"].mean().sort_values(ascending=False)
    )
    for cls in class_counts.head(5).index:
        if cls in class_unchanged:
            print(f"{cls:<15}: {class_unchanged[cls]*100:.4f}% unchanged")

    # 3.2 Feature Extraction for Importance Analysis
    # We extract meta-features from the text to see what predicts the 'class'
    # Subsample for speed (100k samples)
    SAMPLE_SIZE = 100000
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        df_sample = df.copy()

    # Create meta-features
    df_sample["num_digits"] = df_sample["before"].apply(
        lambda x: sum(c.isdigit() for c in x)
    )
    df_sample["num_alpha"] = df_sample["before"].apply(
        lambda x: sum(c.isalpha() for c in x)
    )
    df_sample["num_punct"] = df_sample["before"].apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in x)
    )
    df_sample["is_title"] = df_sample["before"].apply(lambda x: 1 if x.istitle() else 0)
    df_sample["is_upper"] = df_sample["before"].apply(lambda x: 1 if x.isupper() else 0)

    features = [
        "len_before",
        "num_digits",
        "num_alpha",
        "num_punct",
        "is_title",
        "is_upper",
    ]
    target = "class"

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_sample[target])
    X = df_sample[features]

    # 3.3 Correlation (Structured Relationship)
    # Correlation between input length and output length
    len_corr = df_sample["len_before"].corr(df_sample["len_after"])
    print(f"\nCorrelation between Input Length and Output Length: {len_corr:.4f}")

    # Correlation among meta-features
    print("\nMeta-Feature Correlation Matrix (Pearson):")
    corr_mat = X.corr()
    print(corr_mat.round(4))

    # Check for redundancy (Collinear pairs > 0.90)
    print("\nRedundant Feature Pairs (Correlation > 0.90):")
    found_redundancy = False
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i, j]) > 0.90:
                print(
                    f"  {corr_mat.columns[i]} - {corr_mat.columns[j]}: {corr_mat.iloc[i, j]:.4f}"
                )
                found_redundancy = True
    if not found_redundancy:
        print("  None found.")

    # 3.4 Feature Importance (Random Forest)
    print("\nTraining Lightweight Random Forest for Feature Importance...")
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, verbose=0
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top Features for Predicting Token Class:")
    for f in range(len(features)):
        print(f"  {features[indices[f]]:<15}: {importances[indices[f]]:.4f}")

    print("\nSECTION 4: METADATA RELATIONSHIPS (SENTENCE CONTEXT)")
    print("-" * 30)

    # Analyze sentence lengths (number of tokens per sentence)
    # Group by sentence_id and count tokens
    # Note: Using the full df here, not the sample
    sentence_lengths = df.groupby("sentence_id").size()

    print("Sentence Length Distribution (Tokens per Sentence):")
    print(f"Mean: {sentence_lengths.mean():.4f}")
    print(f"Std:  {sentence_lengths.std():.4f}")
    print(f"Min:  {sentence_lengths.min()}")
    print(f"Max:  {sentence_lengths.max()}")

    # Check if sentence length correlates with having 'difficult' classes
    # Define 'difficult' as classes that are not PLAIN or PUNCT
    # We'll do this on the sample to save time if needed, but full df is better for aggregation
    # To keep it fast, we use the already computed 'is_unchanged' on full df

    # Calculate percentage of changed tokens per sentence
    # A sentence with high change % might be more complex
    sent_change_rate = df.groupby("sentence_id")["is_unchanged"].apply(
        lambda x: 1 - x.mean()
    )

    # Correlation between sentence length and change rate
    # We need to align the series
    sent_stats = pd.DataFrame(
        {"length": sentence_lengths, "change_rate": sent_change_rate}
    )

    sent_corr = sent_stats["length"].corr(sent_stats["change_rate"])
    print(
        f"\nCorrelation between Sentence Length and Token Change Rate: {sent_corr:.4f}"
    )
    print(
        "(Positive correlation implies longer sentences have proportionally more tokens requiring normalization)"
    )


if __name__ == "__main__":
    main()
