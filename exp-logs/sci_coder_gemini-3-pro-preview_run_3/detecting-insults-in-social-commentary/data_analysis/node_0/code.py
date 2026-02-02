import os
import sys
import numpy as np
import pandas as pd
import warnings
import random
import codecs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def decode_text(text):
    """
    The dataset description mentions unicode-escaped text.
    We attempt to decode it to get accurate character/word counts.
    """
    if pd.isna(text):
        return ""
    try:
        # If it's a string that looks like a python byte literal representation
        # e.g. "Hello\\nWorld", we decode escape sequences.
        return codecs.decode(str(text), "unicode_escape")
    except Exception:
        return str(text)


def main():
    set_seed(42)

    # 1. Load Data
    data_path = "./metadata/train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # Ensure required columns exist
    if "Insult" not in df.columns or "Comment" not in df.columns:
        print("Error: Required columns 'Insult' or 'Comment' missing.")
        return

    # Preprocess text column for analysis
    # We apply decoding to ensure lengths are calculated on the actual content
    df["clean_text"] = df["Comment"].apply(decode_text)

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")
    target_counts = df["Insult"].value_counts()
    target_ratios = df["Insult"].value_counts(normalize=True)

    print(f"Target Variable: Insult")
    print(
        f"Class 0 (Neutral):   {target_counts.get(0, 0)} ({target_ratios.get(0, 0):.4f})"
    )
    print(
        f"Class 1 (Insulting): {target_counts.get(1, 0)} ({target_ratios.get(1, 0):.4f})"
    )

    if abs(target_ratios.get(0, 0) - target_ratios.get(1, 0)) > 0.2:
        print("Observation: The dataset is imbalanced.")
    else:
        print("Observation: The dataset is relatively balanced.")
    print("-" * 30)

    # ==========================================
    # 3. Input Data Analysis (Text Modality)
    # ==========================================
    print("INPUT DATA ANALYSIS (TEXT)")

    # Calculate lengths
    df["char_len"] = df["clean_text"].apply(len)
    df["word_len"] = df["clean_text"].apply(lambda x: len(x.split()))

    # Sequence Lengths
    print("Sequence Lengths (Character Count):")
    print(f"  Mean: {df['char_len'].mean():.4f}")
    print(f"  Std:  {df['char_len'].std():.4f}")
    print(f"  Min:  {df['char_len'].min():.4f}")
    print(f"  Max:  {df['char_len'].max():.4f}")

    print("\nSequence Lengths (Word Count):")
    print(f"  Mean: {df['word_len'].mean():.4f}")
    print(f"  Std:  {df['word_len'].std():.4f}")
    print(f"  Min:  {df['word_len'].min():.4f}")
    print(f"  Max:  {df['word_len'].max():.4f}")

    # Vocabulary Analysis
    # We use a simple CountVectorizer to estimate vocabulary size
    # We'll limit to a reasonable max_features to avoid memory issues if vocab is huge,
    # but for stats we want the full count if possible.
    try:
        vec = CountVectorizer(stop_words="english", min_df=2)
        vec.fit(df["clean_text"])
        vocab_size = len(vec.vocabulary_)
        print(f"\nVocabulary Size (unique tokens, min_df=2): {vocab_size}")
    except ValueError:
        print("\nVocabulary Size: Unable to compute (possibly empty vocabulary).")

    print("-" * 30)

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Unstructured (Meta-Feature) Relationships
    # Correlation between length and target
    corr_char = df["char_len"].corr(df["Insult"])
    corr_word = df["word_len"].corr(df["Insult"])

    print("Meta-Feature Correlations with Target (Insult):")
    print(f"  Character Length Correlation: {corr_char:.4f}")
    print(f"  Word Count Correlation:       {corr_word:.4f}")

    if abs(corr_word) < 0.05:
        print(
            "  Observation: Comment length has negligible linear correlation with being insulting."
        )
    elif corr_word > 0:
        print(
            "  Observation: Longer comments show a slight positive correlation with being insulting."
        )
    else:
        print(
            "  Observation: Shorter comments show a slight positive correlation with being insulting."
        )

    # Structured/Content Importance
    # We train a lightweight Random Forest on TF-IDF features to find top keywords
    print("\nTop Predictive Features (TF-IDF + Random Forest):")

    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    X_tfidf = tfidf.fit_transform(df["clean_text"])
    y = df["Insult"]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_tfidf, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = np.array(tfidf.get_feature_names_out())

    top_n = 10
    print(f"  Top {top_n} tokens distinguishing classes:")
    for i in range(top_n):
        if i < len(indices):
            print(
                f"    {i+1}. {feature_names[indices[i]]} (Imp: {importances[indices[i]]:.4f})"
            )

    print("-" * 30)


if __name__ == "__main__":
    main()
