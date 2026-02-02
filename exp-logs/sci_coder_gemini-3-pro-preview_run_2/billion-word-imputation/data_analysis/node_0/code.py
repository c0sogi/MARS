import os
import pandas as pd
import numpy as np
import random
import warnings
from collections import Counter
from scipy.stats import skew, kurtosis, pearsonr

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
SEED = 42
SAMPLE_SIZE = 1000000  # 1 Million rows for robust, fast analysis
DATA_PATH = "./metadata/train.parquet"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def suppress_warnings():
    warnings.filterwarnings("ignore")


def load_data(path, sample_size):
    """
    Loads the parquet file. Since the dataset is large (~24M rows),
    we sample a subset for heavy NLP analysis to ensure runtime < 1hr.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    # Read the file
    # We use pandas read_parquet. PyArrow is efficient.
    # To sample without loading everything, we might need a different approach,
    # but with 220GB RAM, we can load columns lazily or just load all and sample.
    # Loading 24M text rows is feasible in memory.

    df = pd.read_parquet(path, columns=["sentence"])

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)

    return df


def get_outlier_count(series):
    """Calculates count of outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()


def analyze_target_distribution(all_words):
    """
    Analyzes the 'target' variable, which in a masked-language task
    is the distribution of words (the vocabulary).
    """
    word_counts = Counter(all_words)
    total_words = len(all_words)
    vocab_size = len(word_counts)

    # Sort by frequency
    sorted_counts = word_counts.most_common()
    frequencies = np.array([count for word, count in sorted_counts])

    # Calculate imbalance/skew
    # In text, this is expected to be Zipfian (highly skewed)
    # We report the coverage of the top 1% of words
    top_1_percent_idx = int(vocab_size * 0.01)
    top_1_percent_counts = frequencies[:top_1_percent_idx].sum()
    coverage_ratio = top_1_percent_counts / total_words

    print("DATA INTEGRITY")
    print(
        f"Analysis performed on training set sample of size: {len(all_words)} words (derived from {SAMPLE_SIZE} sentences)."
    )
    print("-" * 30)

    print("TARGET VARIABLE ANALYSIS")
    print(
        f"Target Type: Text Generation / Self-Supervised Classification (Vocabulary Prediction)"
    )
    print(f"Total Word Count (Target Instances): {total_words}")
    print(f"Vocabulary Size (Unique Classes): {vocab_size}")
    print(
        f"Top 1 Percent Vocabulary Coverage: {coverage_ratio * 100:.4f}% (Indicates extreme class imbalance/Zipfian distribution)"
    )

    # Skewness of the frequency distribution itself
    freq_skew = skew(frequencies)
    freq_kurt = kurtosis(frequencies)
    print(f"Word Frequency Skewness: {freq_skew:.4f}")
    print(f"Word Frequency Kurtosis: {freq_kurt:.4f}")

    print("Top 5 Most Frequent Words (Classes):")
    for word, count in sorted_counts[:5]:
        print(f"  '{word}': {count} ({count/total_words*100:.4f}%)")
    print("-" * 30)


def analyze_input_data(df):
    """
    Analyzes the input text data: Lengths, Character counts, etc.
    """
    # Feature Engineering for Analysis
    df["char_len"] = df["sentence"].str.len()
    # Simple whitespace split for word count
    df["word_count"] = df["sentence"].str.split().str.len()
    df["avg_word_len"] = df["char_len"] / df["word_count"].replace(
        0, 1
    )  # Avoid div by zero

    print("INPUT DATA ANALYSIS (TEXT MODALITY)")

    # 1. Sequence Lengths (Word Counts)
    wc = df["word_count"]
    print("Sequence Lengths (Word Counts):")
    print(f"  Mean: {wc.mean():.4f}")
    print(f"  Std:  {wc.std():.4f}")
    print(f"  Min:  {wc.min():.4f}")
    print(f"  Max:  {wc.max():.4f}")
    print(f"  Outliers (IQR method): {get_outlier_count(wc)}")

    # 2. Character Lengths
    cl = df["char_len"]
    print("Sequence Lengths (Character Counts):")
    print(f"  Mean: {cl.mean():.4f}")
    print(f"  Std:  {cl.std():.4f}")
    print(f"  Min:  {cl.min():.4f}")
    print(f"  Max:  {cl.max():.4f}")

    # 3. Vocabulary (OOV Potential)
    # We already analyzed the vocab in the target section, but here we look at sentence level
    # Check for empty sentences
    empty_sentences = (df["char_len"] == 0).sum()
    print(f"Empty Sentences: {empty_sentences}")

    print("-" * 30)
    return df


def analyze_relationships(df):
    """
    Analyzes relationships between meta-features.
    """
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Correlation between Sentence Length (Words) and Average Word Length
    # Hypothesis: Longer sentences might use more complex (longer) words, or simpler structure?
    corr_len_complexity, _ = pearsonr(df["word_count"], df["avg_word_len"])

    print("Structured Relationships:")
    print(f"  Correlation (Word Count vs. Avg Word Length): {corr_len_complexity:.4f}")

    # Check if longer sentences are just repeats (redundancy check proxy)
    # We can't easily check semantic redundancy without embeddings, but we can check
    # if character length scales perfectly linearly with word count (it should, mostly).
    corr_char_word, _ = pearsonr(df["char_len"], df["word_count"])
    print(f"  Correlation (Char Count vs. Word Count): {corr_char_word:.4f}")

    if corr_char_word > 0.90:
        print(
            "  Observation: Character count and Word count are highly collinear (Redundant features)."
        )

    print("-" * 30)


def main():
    set_seed(SEED)
    suppress_warnings()

    # 1. Load Data
    try:
        df = load_data(DATA_PATH, SAMPLE_SIZE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Pre-processing for Target Analysis
    # Flatten sentences to get the 'corpus' of words
    # We use a simple split to approximate tokens.
    # In a real pipeline, a specific tokenizer (like BERT's) would be used.
    all_words = [word for sentence in df["sentence"] for word in sentence.split()]

    # 3. Target Variable Analysis
    analyze_target_distribution(all_words)

    # 4. Input Data Analysis
    df_analyzed = analyze_input_data(df)

    # 5. Feature Relationships
    analyze_relationships(df_analyzed)


if __name__ == "__main__":
    main()
