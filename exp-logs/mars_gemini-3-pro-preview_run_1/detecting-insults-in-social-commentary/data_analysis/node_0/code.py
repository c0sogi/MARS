import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_custom_date(date_str):
    """Parses date format YYYYMMDDhhmmssZ"""
    if pd.isna(date_str) or date_str == "":
        return None
    try:
        # Remove the trailing 'Z' and parse
        clean_date = str(date_str).replace("Z", "")
        return datetime.strptime(clean_date, "%Y%m%d%H%M%S")
    except (ValueError, TypeError):
        return None


def main():
    set_seed(42)

    # 1. Load Data
    data_path = "./metadata/train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # Ensure strictly training data analysis
    # (The metadata/train.csv is already the 80% split, so we use it directly)

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # =========================================================
    # 2. Target Variable Analysis
    # =========================================================
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 25)

    target_col = "Insult"
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return

    counts = df[target_col].value_counts()
    total = len(df)
    ratio_0 = counts.get(0, 0) / total
    ratio_1 = counts.get(1, 0) / total

    print(f"Target Label: '{target_col}'")
    print(f"Total Samples: {total}")
    print(f"Class 0 (Neutral): {counts.get(0, 0)} ({ratio_0:.4f})")
    print(f"Class 1 (Insult):  {counts.get(1, 0)} ({ratio_1:.4f})")

    # Imbalance check
    minority_class_ratio = min(ratio_0, ratio_1)
    print(f"Minority Class Ratio: {minority_class_ratio:.4f}")
    if minority_class_ratio < 0.2:
        print("Observation: The dataset shows moderate to high class imbalance.")
    else:
        print("Observation: The dataset is relatively balanced.")

    # =========================================================
    # 3. Input Data Analysis (Text Modality)
    # =========================================================
    print("\nINPUT DATA ANALYSIS (TEXT)")
    print("-" * 25)

    text_col = "Comment"

    # Fill NaNs with empty string for analysis
    df[text_col] = df[text_col].fillna("")

    # A. Length Analysis
    # Character counts
    df["char_count"] = df[text_col].apply(len)
    # Word counts (splitting by whitespace)
    df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))

    print("Sequence Length Statistics:")
    print(
        f"  Mean Char Count: {df['char_count'].mean():.4f} (Std: {df['char_count'].std():.4f})"
    )
    print(f"  Min Char Count:  {df['char_count'].min()}")
    print(f"  Max Char Count:  {df['char_count'].max()}")

    print(
        f"  Mean Word Count: {df['word_count'].mean():.4f} (Std: {df['word_count'].std():.4f})"
    )
    print(f"  Min Word Count:  {df['word_count'].min()}")
    print(f"  Max Word Count:  {df['word_count'].max()}")

    # B. Vocabulary Analysis
    # Using CountVectorizer to estimate vocabulary size
    # We use a simple regex token pattern to keep words
    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_features=None)
    try:
        dtm = vectorizer.fit_transform(df[text_col])
        vocab_size = len(vectorizer.vocabulary_)
        print(f"  Vocabulary Size (min_df=2, stop_words='english'): {vocab_size}")

        # Check for OOV potential (rare words)
        # Words that appear only once (we re-run without min_df to check full tail)
        vec_full = CountVectorizer(stop_words="english")
        vec_full.fit(df[text_col])
        full_vocab_size = len(vec_full.vocabulary_)
        rare_ratio = (
            (full_vocab_size - vocab_size) / full_vocab_size
            if full_vocab_size > 0
            else 0
        )
        print(f"  Total Unique Tokens: {full_vocab_size}")
        print(f"  Rare Token Ratio (<2 occurrences): {rare_ratio:.4f}")

    except ValueError:
        print(
            "  Vocabulary Analysis: Unable to build vocabulary (empty text or stop words only)."
        )

    # =========================================================
    # 4. Feature/Signal Relationships
    # =========================================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 25)

    # Feature Engineering for Relationship Analysis
    # We create "Meta-Features" to analyze structured relationships

    # 1. Caps Ratio (Insults often use CAPS)
    def get_caps_ratio(text):
        if len(text) == 0:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)

    df["caps_ratio"] = df[text_col].apply(get_caps_ratio)

    # 2. Exclamation Count
    df["exclam_count"] = df[text_col].apply(lambda x: x.count("!"))

    # 3. Date Features
    # Parse date
    df["datetime"] = df["Date"].apply(parse_custom_date)
    # Create a binary flag for missing date
    df["date_missing"] = df["datetime"].isna().astype(int)
    # Extract hour if date exists, else -1
    df["hour"] = df["datetime"].apply(lambda x: x.hour if x else -1)

    # Prepare tabular data for correlation/importance
    meta_features = [
        "char_count",
        "word_count",
        "caps_ratio",
        "exclam_count",
        "hour",
        "date_missing",
    ]

    # A. Correlation Analysis
    print("Correlation with Target (Insult):")
    correlations = (
        df[meta_features + [target_col]]
        .corr(method="pearson")[target_col]
        .drop(target_col)
    )
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # B. Feature Importance (Lightweight Random Forest)
    print("\nMeta-Feature Importance (Random Forest):")
    X_meta = df[meta_features].fillna(0)
    y = df[target_col]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_meta, y)

    importances = pd.Series(rf.feature_importances_, index=meta_features).sort_values(
        ascending=False
    )
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    # C. Unstructured Relationship (Meta-Feature vs Target Insight)
    print("\nUnstructured/Meta-Feature Insights:")
    # Check if longer comments are more likely to be insults
    avg_len_0 = df[df[target_col] == 0]["char_count"].mean()
    avg_len_1 = df[df[target_col] == 1]["char_count"].mean()
    print(f"  Avg Char Length (Neutral):   {avg_len_0:.4f}")
    print(f"  Avg Char Length (Insulting): {avg_len_1:.4f}")

    if avg_len_1 > avg_len_0:
        print("  -> Insulting comments tend to be longer on average.")
    else:
        print("  -> Insulting comments tend to be shorter on average.")

    # Check Caps Lock usage
    avg_caps_0 = df[df[target_col] == 0]["caps_ratio"].mean()
    avg_caps_1 = df[df[target_col] == 1]["caps_ratio"].mean()
    print(f"  Avg Caps Ratio (Neutral):    {avg_caps_0:.4f}")
    print(f"  Avg Caps Ratio (Insulting):  {avg_caps_1:.4f}")

    # Check Exclamations
    avg_exc_0 = df[df[target_col] == 0]["exclam_count"].mean()
    avg_exc_1 = df[df[target_col] == 1]["exclam_count"].mean()
    print(f"  Avg Exclamations (Neutral):   {avg_exc_0:.4f}")
    print(f"  Avg Exclamations (Insulting): {avg_exc_1:.4f}")


if __name__ == "__main__":
    main()
