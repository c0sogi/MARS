import pandas as pd
import numpy as np
import os
import random
import warnings
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) - len(c)) == 0:
        return 0.0
    return float(len(c)) / (len(a) + len(b) - len(c))


def main():
    set_seed(42)

    # 1. Data Loading and Integrity
    # Using metadata/train.csv as per instructions to prevent leakage
    data_path = "./metadata/train.csv"

    # Fallback safety check (though metadata is guaranteed by prompt)
    if not os.path.exists(data_path):
        data_path = "./input/train.csv"

    df = pd.read_csv(data_path)

    # Basic cleaning for analysis (ensure strings)
    df.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)
    df["text"] = df["text"].astype(str)
    df["selected_text"] = df["selected_text"].astype(str)

    print("DATA INTEGRITY CHECK")
    print(f"Analysis performed on training set with {len(df)} samples.")
    print("-" * 30)

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("\nTARGET VARIABLE ANALYSIS")

    # In this task, 'sentiment' is the primary conditioning variable (class).
    # We analyze its distribution.
    sentiment_counts = df["sentiment"].value_counts()
    total = len(df)

    print("Distribution (Sentiment):")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/total:.4f})")

    # Imbalance
    max_count = sentiment_counts.max()
    min_count = sentiment_counts.min()
    print(f"\nClass Balance Ratio (Max/Min): {max_count/min_count:.4f}")

    # ==========================================
    # 3. Input Data Analysis (Text Modality)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # Compute lengths
    df["char_len"] = df["text"].apply(len)
    df["word_len"] = df["text"].apply(lambda x: len(x.split()))

    # Length Statistics
    print("Sequence Lengths (Input Text):")
    print(
        f"  Character Count: Mean={df['char_len'].mean():.4f}, Std={df['char_len'].std():.4f}, Min={df['char_len'].min()}, Max={df['char_len'].max()}"
    )
    print(
        f"  Word Count:      Mean={df['word_len'].mean():.4f}, Std={df['word_len'].std():.4f}, Min={df['word_len'].min()}, Max={df['word_len'].max()}"
    )

    # Vocabulary Statistics
    # Tokenize simply by splitting on whitespace for EDA
    all_tokens = [word for text in df["text"] for word in text.split()]
    vocab_counter = Counter(all_tokens)
    vocab_size = len(vocab_counter)

    print("\nVocabulary Stats:")
    print(f"  Unique Vocabulary Size: {vocab_size}")

    # Rare words (appearing only once)
    rare_words = sum(1 for c in vocab_counter.values() if c == 1)
    print(
        f"  Rare Words (<2 freq):   {rare_words} ({rare_words/vocab_size:.4f} of vocab)"
    )

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Generate Meta-Features
    # Target length (Word count of the selected text)
    df["target_word_len"] = df["selected_text"].apply(lambda x: len(x.split()))

    # Jaccard Similarity (Input vs Target)
    # This measures how much of the original text is selected
    df["jaccard"] = df.apply(lambda x: jaccard(x["text"], x["selected_text"]), axis=1)

    print("Unstructured (Meta-Feature) Relationships:")

    # 1. Relationship between Sentiment and Overlap (Jaccard)
    # This is critical for this dataset: Neutral usually has Jaccard ~ 1.0 (Text == Selected Text)
    print("  Mean Jaccard Score by Sentiment:")
    jaccard_stats = df.groupby("sentiment")["jaccard"].mean()
    for sentiment, score in jaccard_stats.items():
        print(f"    {sentiment}: {score:.4f}")

    # 2. Relationship between Input Length and Target Length
    # Do longer tweets result in longer selected text spans?
    corr_len = df["word_len"].corr(df["target_word_len"])
    print(f"\n  Correlation (Input Word Len vs Target Word Len): {corr_len:.4f}")

    # 3. Relationship between Sentiment and Input Length
    # Are negative/positive tweets longer or shorter than neutral ones?
    print("\n  Mean Input Word Length by Sentiment:")
    len_stats = df.groupby("sentiment")["word_len"].mean()
    for sentiment, length in len_stats.items():
        print(f"    {sentiment}: {length:.4f}")


if __name__ == "__main__":
    main()
