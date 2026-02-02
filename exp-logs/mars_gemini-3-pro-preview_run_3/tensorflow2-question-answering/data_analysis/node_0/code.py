import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
TRAIN_FILE = "simplified-nq-train.jsonl"
SAMPLE_SIZE = 10000  # Number of samples for heavy text analysis
RANDOM_SEED = 42

# Set seeds
np.random.seed(RANDOM_SEED)


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return None
    return pd.read_csv(METADATA_PATH)


def get_text_stats(metadata_df, sample_size):
    """
    Reads a sample of the training data to compute text statistics.
    Returns a DataFrame with meta-features and a vocabulary counter.
    """
    if len(metadata_df) > sample_size:
        sample_df = metadata_df.sample(n=sample_size, random_state=RANDOM_SEED).copy()
    else:
        sample_df = metadata_df.copy()

    file_path = os.path.join(INPUT_DIR, TRAIN_FILE)

    doc_lengths = []
    q_lengths = []
    q_starters = []
    vocab = Counter()

    # Efficiently read specific lines using byte offsets
    with open(file_path, "rb") as f:
        for _, row in sample_df.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))

                # Document Text Analysis
                doc_text = data.get("document_text", "")
                doc_tokens = doc_text.split()
                doc_len = len(doc_tokens)
                doc_lengths.append(doc_len)

                # Question Text Analysis
                q_text = data.get("question_text", "")
                q_tokens = q_text.split()
                q_len = len(q_tokens)
                q_lengths.append(q_len)

                # Question Starter (Meta-feature)
                if q_tokens:
                    q_starters.append(q_tokens[0].lower())
                else:
                    q_starters.append("unknown")

                # Update Vocab (lightweight, just unigrams from sample)
                # We limit update to avoid memory explosion in this script
                vocab.update(q_tokens)
                # Sampling doc tokens for vocab to save time/memory
                if len(doc_tokens) > 100:
                    vocab.update(doc_tokens[:100])
                else:
                    vocab.update(doc_tokens)

            except json.JSONDecodeError:
                continue

    sample_df["doc_word_count"] = doc_lengths
    sample_df["q_word_count"] = q_lengths
    sample_df["q_start_word"] = q_starters

    return sample_df, vocab


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_col = "stratify_label"
    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Target Variable: {target_col} (Proxy for Answer Type)")
    print(f"Total Samples: {total}")
    print("Class Distribution:")
    for label, count in counts.items():
        ratio = count / total
        print(f"  {label}: {count} ({ratio:.4f})")

    # Calculate imbalance ratio (Majority / Minority)
    if len(counts) > 1:
        imbalance = counts.max() / counts.min()
        print(f"Class Imbalance Ratio (Max/Min): {imbalance:.4f}")
    print("")


def analyze_text_data(df, vocab):
    print("INPUT DATA ANALYSIS (TEXT MODALITY)")
    print("-" * 30)

    # Sequence Lengths
    print("Sequence Lengths (Word Counts):")

    for col, name in [("doc_word_count", "Document"), ("q_word_count", "Question")]:
        stats = df[col].describe()
        iqr = stats["75%"] - stats["25%"]
        upper_bound = stats["75%"] + 1.5 * iqr
        outliers = df[df[col] > upper_bound].shape[0]

        print(f"  {name} Lengths:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std Dev: {stats['std']:.4f}")
        print(f"    Min: {stats['min']:.4f}")
        print(f"    Max: {stats['max']:.4f}")
        print(f"    Outlier Count (IQR method): {outliers}")

    # Vocabulary
    print("\nVocabulary Statistics (Sampled):")
    print(f"  Unique Tokens Observed: {len(vocab)}")
    print(f"  Most Common Tokens: {', '.join([x[0] for x in vocab.most_common(5)])}")

    # Check for OOV potential (hapax legomena ratio in sample)
    singletons = sum(1 for count in vocab.values() if count == 1)
    print(f"  Singleton Ratio (Potential OOV): {singletons/len(vocab):.4f}")
    print("")


def analyze_relationships(df):
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # 1. Unstructured (Meta-Feature) Relationships
    print("Meta-Feature vs Target Relationships:")
    # Group by target and get mean lengths
    grouped = df.groupby("stratify_label")[["doc_word_count", "q_word_count"]].mean()
    print("  Average Lengths by Answer Type:")
    print(grouped.to_string(float_format="{:.4f}".format))

    # 2. Structured Relationships (Correlation)
    print("\nStructured Relationships:")
    corr = df[["doc_word_count", "q_word_count"]].corr(method="pearson").iloc[0, 1]
    print(f"  Correlation (Doc Length vs Question Length): {corr:.4f}")
    if abs(corr) > 0.90:
        print(
            "  Redundancy Flag: High collinearity detected between Doc Length and Question Length."
        )
    else:
        print("  Redundancy Flag: No high collinearity detected.")

    # 3. Feature Importance (Random Forest)
    print("\nFeature Importance (Predicting Answer Type):")

    # Prepare data
    # Encode categorical 'q_start_word'
    le = LabelEncoder()
    # Keep top 20 start words, map rest to 'other' to avoid high cardinality noise
    top_starters = df["q_start_word"].value_counts().nlargest(20).index
    df["q_start_word_encoded"] = df["q_start_word"].apply(
        lambda x: x if x in top_starters else "other"
    )
    df["q_start_word_encoded"] = le.fit_transform(df["q_start_word_encoded"])

    features = ["doc_word_count", "q_word_count", "q_start_word_encoded"]
    X = df[features]
    y = df["stratify_label"]

    # Train lightweight RF
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )

    print("  Top Predictive Meta-Features:")
    for feat, imp in importances.head(5).items():
        print(f"    {feat}: {imp:.4f}")
    print("")


def main():
    # 1. Data Integrity & Loading
    print("LOADING METADATA...")
    metadata = load_metadata()
    if metadata is None:
        return

    # 2. Target Variable Analysis (on full metadata)
    analyze_target(metadata)

    # 3. Input Data Analysis (on sample)
    print(f"SAMPLING {SAMPLE_SIZE} EXAMPLES FOR TEXT ANALYSIS...")
    sample_df, vocab = get_text_stats(metadata, SAMPLE_SIZE)

    analyze_text_data(sample_df, vocab)

    # 4. Feature Relationships (on sample)
    analyze_relationships(sample_df)

    print("EDA COMPLETED.")


if __name__ == "__main__":
    main()
