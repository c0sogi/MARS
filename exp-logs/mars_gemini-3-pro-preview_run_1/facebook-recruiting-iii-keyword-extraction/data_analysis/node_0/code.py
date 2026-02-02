import pandas as pd
import numpy as np
import os
import re
import gc
import random
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_eda():
    set_seed(42)

    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")

    print("--- 1. DATA LOADING ---")

    if not os.path.exists(TRAIN_META_PATH):
        raise FileNotFoundError(f"{TRAIN_META_PATH} not found.")

    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"{TRAIN_CSV_PATH} not found.")

    # Load Metadata
    print(f"Reading metadata from {TRAIN_META_PATH}...")
    df_meta = pd.read_csv(TRAIN_META_PATH)
    print(f"Metadata Rows: {len(df_meta)}")

    # Load Train Data
    # We load Id, Title, Body. Tags are in metadata, but train.csv has them too.
    # We rely on metadata for the split, but we need Title and Body from train.csv.
    print(f"Reading text data from {TRAIN_CSV_PATH}...")
    try:
        # Loading full dataset might be heavy, but we have 220GB RAM.
        # We use usecols to minimize memory usage.
        df_text = pd.read_csv(
            TRAIN_CSV_PATH,
            usecols=["Id", "Title", "Body"],
            dtype={"Id": "int64", "Title": "object", "Body": "object"},
        )
    except Exception as e:
        print(f"Error reading train.csv: {e}")
        return

    # Merge
    print("Merging metadata with text data...")
    df = df_meta.merge(df_text, on="Id", how="inner")

    # Clean up memory
    del df_meta, df_text
    gc.collect()

    print(f"Final Training Set Shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("\n--- 2. TARGET VARIABLE ANALYSIS ---")

    # Ensure Tags is string
    df["Tags"] = df["Tags"].fillna("").astype(str)

    # Calculate number of tags per question
    df["num_tags"] = df["Tags"].apply(lambda x: len(x.strip().split()))

    print("Distribution of Target Variable (Number of Tags per Question):")
    desc = df["num_tags"].describe()
    print(f"Mean: {desc['mean']:.4f}")
    print(f"Std:  {desc['std']:.4f}")
    print(f"Min:  {desc['min']:.4f}")
    print(f"Max:  {desc['max']:.4f}")

    # Tag Frequency Analysis
    print("\nTag Frequency Analysis:")
    all_tags = df["Tags"].str.split().explode()
    tag_counts = all_tags.value_counts()

    print(f"Total Unique Tags: {len(tag_counts)}")
    print("Top 10 Most Frequent Tags:")
    print(tag_counts.head(10).to_string())

    # Rare labels analysis (< 1% frequency in dataset)
    # We calculate frequency relative to the total number of samples
    tag_support = tag_counts / len(df)
    rare_tags = tag_support[tag_support < 0.01]

    print(f"\nNumber of Rare Tags (< 1% sample frequency): {len(rare_tags)}")
    print(
        f"Percentage of tags that are rare: {(len(rare_tags)/len(tag_counts))*100:.4f}%"
    )

    # ---------------------------------------------------------
    # 3. INPUT DATA ANALYSIS (TEXT)
    # ---------------------------------------------------------
    print("\n--- 3. INPUT DATA ANALYSIS (TEXT) ---")

    # Sample for text stats to ensure runtime < 1 hour
    # Processing millions of HTML bodies is slow
    SAMPLE_SIZE = 200000
    if len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} random rows for detailed text analysis...")
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        df_sample = df.copy()

    # Title Analysis
    print("\n[Title Analysis]")
    # Vectorized string operations
    df_sample["title_char_len"] = df_sample["Title"].fillna("").str.len()
    df_sample["title_word_len"] = df_sample["Title"].fillna("").str.split().str.len()

    print("Title Character Lengths:")
    print(
        f"Mean: {df_sample['title_char_len'].mean():.4f}, Std: {df_sample['title_char_len'].std():.4f}"
    )
    print(
        f"Min: {df_sample['title_char_len'].min():.4f}, Max: {df_sample['title_char_len'].max():.4f}"
    )

    print("Title Word Lengths:")
    print(
        f"Mean: {df_sample['title_word_len'].mean():.4f}, Std: {df_sample['title_word_len'].std():.4f}"
    )

    # Body Analysis
    print("\n[Body Analysis]")

    # Pre-compile regex for HTML stripping
    cleanr = re.compile(r"<[^>]+>")

    def process_body(text):
        if not isinstance(text, str):
            return 0, 0
        # Strip HTML tags
        text = cleanr.sub(" ", text)
        # Char len (approx)
        c_len = len(text)
        # Word len (simple whitespace split)
        w_len = len(text.split())
        return c_len, w_len

    # Use list comprehension for speed
    body_data = [process_body(t) for t in df_sample["Body"]]
    df_sample["body_char_len"] = [x[0] for x in body_data]
    df_sample["body_word_len"] = [x[1] for x in body_data]

    print("Body Character Lengths (HTML Stripped):")
    print(
        f"Mean: {df_sample['body_char_len'].mean():.4f}, Std: {df_sample['body_char_len'].std():.4f}"
    )
    print(
        f"Min: {df_sample['body_char_len'].min():.4f}, Max: {df_sample['body_char_len'].max():.4f}"
    )

    print("Body Word Lengths (HTML Stripped):")
    print(
        f"Mean: {df_sample['body_word_len'].mean():.4f}, Std: {df_sample['body_word_len'].std():.4f}"
    )

    # Vocabulary Analysis
    print("\n[Vocabulary Analysis]")
    # We'll use a Counter on the sampled data
    vocab = Counter()

    # Update with Title words
    for t in df_sample["Title"].fillna(""):
        vocab.update(t.lower().split())

    # Update with Body words (cleaning on the fly)
    for b in df_sample["Body"].fillna(""):
        t = cleanr.sub(" ", b)
        vocab.update(t.lower().split())

    print(f"Estimated Vocabulary Size (on {SAMPLE_SIZE} samples): {len(vocab)}")
    print(
        f"OOV Potential: {len([w for w, c in vocab.items() if c == 1])} words appear only once in sample."
    )

    # ---------------------------------------------------------
    # 4. FEATURE/SIGNAL RELATIONSHIPS
    # ---------------------------------------------------------
    print("\n--- 4. FEATURE/SIGNAL RELATIONSHIPS ---")

    # Correlation
    # We correlate lengths with num_tags (meta-feature relationship)
    feats = [
        "title_char_len",
        "title_word_len",
        "body_char_len",
        "body_word_len",
        "num_tags",
    ]
    corr_matrix = df_sample[feats].corr(method="pearson")

    print("Correlation with Target (Num Tags):")
    print(corr_matrix["num_tags"].drop("num_tags").to_string())

    # Feature Importance
    print("\n[Meta-Feature Importance]")
    # Train a small Random Forest to predict num_tags from lengths
    X = df_sample[
        ["title_char_len", "body_char_len", "title_word_len", "body_word_len"]
    ].fillna(0)
    y = df_sample["num_tags"]

    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    print("Top Features predicting Num Tags:")
    imps = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print(imps.to_string())

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    run_eda()
