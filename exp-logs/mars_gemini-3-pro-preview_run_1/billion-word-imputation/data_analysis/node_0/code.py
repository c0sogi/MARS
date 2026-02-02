import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import skew, kurtosis
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = (
    500000  # 500k samples is statistically sufficient and fits well within time limits
)
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_outlier_count(data):
    """Calculates outlier count using IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.sum((data < lower_bound) | (data > upper_bound))


def analyze_text_data():
    print("DATA INTEGRITY")
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load data
    # We read the ID and sentence. We assume the 'sentence' column exists based on metadata generation.
    try:
        df = pd.read_csv(METADATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    total_rows = len(df)
    print(f"Total rows in training set: {total_rows}")

    # Check for nulls in sentence column
    null_sentences = df["sentence"].isnull().sum()
    if null_sentences > 0:
        print(f"Rows with missing sentence text: {null_sentences}")
        df = df.dropna(subset=["sentence"])

    # Sampling for expensive text operations
    if len(df) > SAMPLE_SIZE:
        print(
            f"Dataset too large for rapid detailed text analysis. Sampling {SAMPLE_SIZE} rows using fixed seed."
        )
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()
    else:
        df_sample = df.copy()

    print(f"Analysis performed on {len(df_sample)} samples.")
    print("-" * 30)

    # Preprocessing: Simple whitespace tokenization
    # The dataset description implies pre-tokenized text (spaces around punctuation).
    # We create a list of lists of tokens.
    sentences = df_sample["sentence"].astype(str).tolist()
    tokenized_sentences = [s.split() for s in sentences]

    # Flatten for vocabulary analysis
    all_tokens = [token for sublist in tokenized_sentences for token in sublist]
    vocab_counter = Counter(all_tokens)
    vocab_counts = list(vocab_counter.values())

    print("TARGET VARIABLE ANALYSIS")
    # In a Cloze task (Fill-Mask), the 'target' is the missing word.
    # The distribution of targets corresponds to the frequency of words in the corpus.
    # This is effectively a Multi-class classification with |Vocab| classes.

    print(
        "Task Type: Self-Supervised Text Infilling (treated as Classification over Vocabulary)"
    )

    if len(vocab_counts) > 0:
        # Distribution statistics of word frequencies
        mean_freq = np.mean(vocab_counts)
        std_freq = np.std(vocab_counts)
        max_freq = np.max(vocab_counts)
        min_freq = np.min(vocab_counts)

        print(f"Word Frequency Mean: {mean_freq:.4f}")
        print(f"Word Frequency Std: {std_freq:.4f}")

        # Imbalance/Skew
        # Calculate skewness of the frequency distribution (Zipfian distributions are highly skewed)
        freq_skew = skew(vocab_counts)
        freq_kurt = kurtosis(vocab_counts)

        print(f"Word Frequency Skewness: {freq_skew:.4f}")
        print(f"Word Frequency Kurtosis: {freq_kurt:.4f}")

        # Class Balance Ratios
        # Ratio of most frequent (Stop words) to median frequent
        median_freq = np.median(vocab_counts)
        if median_freq == 0:
            median_freq = 1
        balance_ratio = max_freq / median_freq
        print(f"Class Balance Ratio (Max/Median Frequency): {balance_ratio:.4f}")

        # Top classes
        print("Top 5 Most Frequent Words (Classes):")
        for word, count in vocab_counter.most_common(5):
            print(f"  '{word}': {count}")
    else:
        print("No tokens found.")
    print("-" * 30)

    print("INPUT DATA ANALYSIS (TEXT)")

    # Lengths
    # Word counts per sentence
    seq_lens_words = np.array([len(s) for s in tokenized_sentences])
    # Character counts per sentence
    seq_lens_chars = np.array([len(s) for s in sentences])

    print("Sequence Lengths (Words):")
    print(f"  Mean: {np.mean(seq_lens_words):.4f}")
    print(f"  Std:  {np.std(seq_lens_words):.4f}")
    print(f"  Min:  {np.min(seq_lens_words):.4f}")
    print(f"  Max:  {np.max(seq_lens_words):.4f}")
    print(f"  Outliers (IQR): {get_outlier_count(seq_lens_words)}")

    print("Sequence Lengths (Characters):")
    print(f"  Mean: {np.mean(seq_lens_chars):.4f}")
    print(f"  Std:  {np.std(seq_lens_chars):.4f}")

    # Vocabulary
    vocab_size = len(vocab_counter)
    total_tokens = len(all_tokens)

    # OOV Potential: Words that appear only once (Hapax Legomena)
    # These are likely to be OOV in a random test split if not handled.
    singletons = sum(1 for c in vocab_counts if c == 1)
    oov_potential = singletons / vocab_size if vocab_size > 0 else 0

    print("Vocabulary Statistics:")
    print(f"  Unique Vocabulary Size: {vocab_size}")
    print(f"  Total Tokens in Sample: {total_tokens}")
    print(f"  Hapax Legomena (Words appearing once): {singletons}")
    print(f"  OOV Potential (Singleton Ratio): {oov_potential:.4f}")
    print("-" * 30)

    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Unstructured (Meta-Feature) Relationships
    # We analyze the relationship between Sentence Length (Words) and Average Word Length.
    # Hypothesis: Longer sentences might use simpler (shorter) words, or more complex (longer) words?

    # Calculate Average Word Length per sentence
    # Avoid division by zero
    avg_word_lens = []
    for words in tokenized_sentences:
        if len(words) > 0:
            lens = [len(w) for w in words]
            avg_word_lens.append(np.mean(lens))
        else:
            avg_word_lens.append(0)
    avg_word_lens = np.array(avg_word_lens)

    # Create a temporary DataFrame for correlation
    meta_df = pd.DataFrame(
        {
            "n_words": seq_lens_words,
            "n_chars": seq_lens_chars,
            "avg_word_len": avg_word_lens,
        }
    )

    # Correlation Matrix
    corr_matrix = meta_df.corr(method="pearson")

    print("Meta-Feature Correlations (Pearson):")
    print(
        f"  Correlation (Num Words vs Num Chars): {corr_matrix.loc['n_words', 'n_chars']:.4f}"
    )
    print(
        f"  Correlation (Num Words vs Avg Word Len): {corr_matrix.loc['n_words', 'avg_word_len']:.4f}"
    )

    # Interpretation
    r_words_avg = corr_matrix.loc["n_words", "avg_word_len"]
    if abs(r_words_avg) < 0.1:
        rel_text = "No significant linear relationship"
    elif r_words_avg > 0:
        rel_text = "Positive relationship (Longer sentences tend to have longer words)"
    else:
        rel_text = "Negative relationship (Longer sentences tend to have shorter words)"

    print(f"  Insight: {rel_text} between sentence length and word complexity.")

    # Redundancy Check
    if corr_matrix.loc["n_words", "n_chars"] > 0.90:
        print(
            "  Redundancy Flag: 'Num Words' and 'Num Chars' are highly collinear (> 0.90)."
        )


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    analyze_text_data()
