import pandas as pd
import numpy as np
import os
import random
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_FILE = "train.csv"
RAW_TRAIN_FILE = "train.csv"
SEED = 42


def set_seed(seed):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    # ==========================================
    # 1. Data Loading & Integrity
    # ==========================================

    # Load Metadata to define the training set
    meta_path = os.path.join(METADATA_DIR, TRAIN_META_FILE)
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found at {meta_path}")
        return

    # Read metadata (Id and Tags)
    # This file defines the ground truth for the training split
    df_meta = pd.read_csv(meta_path)

    # Load Raw Data (Id, Title, Body)
    # We read the raw file to get the text content.
    raw_path = os.path.join(INPUT_DIR, RAW_TRAIN_FILE)

    # We use 'usecols' to optimize memory usage, loading only what's needed.
    try:
        df_raw = pd.read_csv(raw_path, usecols=["Id", "Title", "Body"])
    except ValueError:
        # Fallback in case columns are named differently (though schema is known)
        df_raw = pd.read_csv(raw_path)

    # Merge to create the final Training DataFrame
    # An inner join on 'Id' ensures we strictly adhere to the metadata's train split
    df = pd.merge(df_meta, df_raw, on="Id", how="inner")

    # Handle missing values for text processing
    df["Tags"] = df["Tags"].fillna("")
    df["Title"] = df["Title"].fillna("")
    df["Body"] = df["Body"].fillna("")

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    # Parse Tags (Space-delimited strings)
    df["tag_list"] = df["Tags"].str.split()

    # Flatten list to get all individual tag occurrences
    all_tags = [tag for tags in df["tag_list"] for tag in tags]
    tag_counts = Counter(all_tags)

    # Statistics
    num_unique_tags = len(tag_counts)
    tags_per_row = df["tag_list"].apply(len)

    print(f"Total Training Samples: {len(df)}")
    print(f"Total Unique Tags: {num_unique_tags}")

    # Distribution of Tags per Question
    print(f"Tags per Question Distribution:")
    print(f"  Mean: {tags_per_row.mean():.4f}")
    print(f"  Std:  {tags_per_row.std():.4f}")
    print(f"  Min:  {tags_per_row.min():.4f}")
    print(f"  Max:  {tags_per_row.max():.4f}")

    # Imbalance / Skew
    if tag_counts:
        most_common = tag_counts.most_common(5)
        print(f"Top 5 Tags: {', '.join([f'{t} ({c})' for t, c in most_common])}")

        # Count rare tags (appearing only once)
        rare_tag_count = sum(1 for c in tag_counts.values() if c == 1)
        print(
            f"Rare Tags (Frequency = 1): {rare_tag_count} ({rare_tag_count/num_unique_tags*100:.2f}% of unique tags)"
        )

    # ==========================================
    # 3. Input Data Analysis (Text)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # --- Length Analysis ---
    # Calculate lengths for Title
    title_char_len = df["Title"].str.len()
    title_word_len = df["Title"].str.split().str.len()

    # Calculate lengths for Body
    body_char_len = df["Body"].str.len()
    body_word_len = df["Body"].str.split().str.len()

    print("Title Lengths (Characters):")
    print(
        f"  Mean: {title_char_len.mean():.4f}, Std: {title_char_len.std():.4f}, Min: {title_char_len.min():.4f}, Max: {title_char_len.max():.4f}"
    )

    print("Body Lengths (Characters):")
    print(
        f"  Mean: {body_char_len.mean():.4f}, Std: {body_char_len.std():.4f}, Min: {body_char_len.min():.4f}, Max: {body_char_len.max():.4f}"
    )

    print("Title Lengths (Words):")
    print(f"  Mean: {title_word_len.mean():.4f}, Max: {title_word_len.max():.4f}")

    print("Body Lengths (Words):")
    print(f"  Mean: {body_word_len.mean():.4f}, Max: {body_word_len.max():.4f}")

    # --- Vocabulary Analysis ---
    # We use a sample for vocabulary estimation to ensure the script runs quickly
    SAMPLE_SIZE = 100000
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        df_sample = df

    print(f"Vocabulary Analysis (Sampled {len(df_sample)} rows):")

    # Title Vocabulary
    # Using CountVectorizer to tokenize and count unique words
    vec_title = CountVectorizer(max_features=None)
    vec_title.fit(df_sample["Title"])
    print(f"  Title Vocabulary Size: {len(vec_title.vocabulary_)}")

    # Body Vocabulary
    # Note: Body contains HTML. Simple tokenization is used here for EDA.
    vec_body = CountVectorizer(max_features=None)
    vec_body.fit(df_sample["Body"])
    print(f"  Body Vocabulary Size:  {len(vec_body.vocabulary_)}")

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # --- Structured Relationships (Meta-features) ---
    # Create a dataframe of meta-features
    meta_df = pd.DataFrame(
        {
            "title_len": title_char_len,
            "body_len": body_char_len,
            "num_tags": tags_per_row,
        }
    )

    # Calculate Pearson Correlation
    corr = meta_df.corr(method="pearson")

    print("Correlations (Pearson):")
    print(f"  Title Length vs Num Tags: {corr.loc['title_len', 'num_tags']:.4f}")
    print(f"  Body Length vs Num Tags:  {corr.loc['body_len', 'num_tags']:.4f}")
    print(f"  Title Length vs Body Length: {corr.loc['title_len', 'body_len']:.4f}")

    # --- Unstructured (Meta-Feature) Relationships ---
    # Analyze if the most common tag is associated with longer/shorter body text
    if tag_counts:
        top_tag = tag_counts.most_common(1)[0][0]

        # Create binary indicator for presence of the top tag
        has_top_tag = df["tag_list"].apply(lambda x: top_tag in x)

        len_with = body_char_len[has_top_tag].mean()
        len_without = body_char_len[~has_top_tag].mean()

        print(f"Relationship with Top Tag '{top_tag}':")
        print(f"  Avg Body Length (With Tag):    {len_with:.4f}")
        print(f"  Avg Body Length (Without Tag): {len_without:.4f}")


if __name__ == "__main__":
    main()
