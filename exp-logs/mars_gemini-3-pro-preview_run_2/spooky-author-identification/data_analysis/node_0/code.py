import pandas as pd
import numpy as np
import os
import random
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Configuration & Setup
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main():
    # Load Data
    # Using the metadata file as specified in the instructions
    train_path = "./metadata/train.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    df = pd.read_csv(train_path)

    # Identify columns
    # Based on dataset description: 'text' is input, 'author' is target
    text_col = "text"
    target_col = "author"

    print("==== TARGET VARIABLE ANALYSIS ====")
    # Distribution & Imbalance
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        total_samples = len(df)
        print(f"Target Variable: '{target_col}'")
        print(f"Total Samples: {total_samples}")
        print("Class Distribution:")
        for cls, count in counts.items():
            ratio = count / total_samples
            print(f"  - {cls}: {count} ({ratio:.4f})")

        # Check for imbalance
        max_ratio = counts.max() / total_samples
        min_ratio = counts.min() / total_samples
        print(f"Imbalance Ratio (Max/Min): {(counts.max()/counts.min()):.4f}")
    else:
        print(f"Target column '{target_col}' not found.")

    print("\n==== INPUT DATA ANALYSIS (TEXT) ====")
    # Text Specific Analysis
    # 1. Lengths
    # Calculate character count and word count
    df["char_count"] = df[text_col].astype(str).apply(len)
    df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    print("Sequence Lengths (Characters):")
    print(f"  - Mean: {df['char_count'].mean():.4f}")
    print(f"  - Std:  {df['char_count'].std():.4f}")
    print(f"  - Min:  {df['char_count'].min():.4f}")
    print(f"  - Max:  {df['char_count'].max():.4f}")

    print("Sequence Lengths (Words):")
    print(f"  - Mean: {df['word_count'].mean():.4f}")
    print(f"  - Std:  {df['word_count'].std():.4f}")
    print(f"  - Min:  {df['word_count'].min():.4f}")
    print(f"  - Max:  {df['word_count'].max():.4f}")

    # 2. Vocabulary
    # Use TfidfVectorizer to build vocabulary (removing English stop words to focus on content)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=None)
    try:
        X_tfidf = vectorizer.fit_transform(df[text_col].astype(str))
        vocab = vectorizer.vocabulary_
        print(f"Vocabulary Size (English Stopwords Removed): {len(vocab)}")

        # OOV Potential proxy: Count terms that appear in very few documents
        # Sum binary occurrence across documents
        term_counts = (X_tfidf > 0).sum(axis=0)
        # Convert to array
        term_counts = np.array(term_counts).flatten()
        rare_terms = np.sum(term_counts == 1)
        print(
            f"Rare Terms (Appearing in only 1 doc): {rare_terms} ({(rare_terms/len(vocab)):.4f} of vocab)"
        )
    except Exception as e:
        print(f"Error during vocabulary analysis: {e}")

    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # 1. Unstructured (Meta-Feature) Relationships
    # Analyze relationship between Length and Author
    print("Meta-Feature Relationship: Word Count vs Author")
    if target_col in df.columns:
        mean_lens = df.groupby(target_col)["word_count"].mean()
        for cls, mlen in mean_lens.items():
            print(f"  - Mean Word Count for {cls}: {mlen:.4f}")

        # Check correlation if we encode target (just for general direction)
        le = LabelEncoder()
        y_encoded = le.fit_transform(df[target_col])
        corr = np.corrcoef(df["word_count"], y_encoded)[0, 1]
        print(f"  - Correlation (Word Count vs Encoded Target): {corr:.4f}")

    # 2. Structured Relationships (Feature Importance)
    # Train a lightweight Random Forest to find top predictive words
    print("Feature Importance: Top 5 Predictive Words (TF-IDF + Random Forest)")

    try:
        # Limit features to speed up RF training
        rf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X_rf = rf_vectorizer.fit_transform(df[text_col].astype(str))
        y_rf = df[target_col]

        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
        )
        rf.fit(X_rf, y_rf)

        importances = rf.feature_importances_
        feature_names = np.array(rf_vectorizer.get_feature_names_out())

        # Get indices of top 5 features
        top_indices = np.argsort(importances)[::-1][:5]

        for idx in top_indices:
            print(f"  - {feature_names[idx]}: {importances[idx]:.4f}")

    except Exception as e:
        print(f"Error during feature importance analysis: {e}")


if __name__ == "__main__":
    main()
