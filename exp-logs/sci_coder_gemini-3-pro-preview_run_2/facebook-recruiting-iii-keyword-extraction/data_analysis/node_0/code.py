import pandas as pd
import numpy as np
import re
import sys
import os
import gc
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_targets(df):
    print_section("Target Variable Analysis")

    # Tags are space-delimited strings.
    # We need to analyze individual tags (multi-label) and tag counts per sample.

    # Drop rows with missing tags if any (though metadata generation handled fillna)
    tags_series = df["Tags"].astype(str)

    # 1. Number of tags per question
    tags_per_row = tags_series.apply(lambda x: len(x.split()))

    print("--- Distribution of Tag Counts per Question ---")
    print(f"Mean tags per question: {tags_per_row.mean():.4f}")
    print(f"Std tags per question:  {tags_per_row.std():.4f}")
    print(f"Min tags per question:  {tags_per_row.min():.4f}")
    print(f"Max tags per question:  {tags_per_row.max():.4f}")

    # 2. Tag Frequency Analysis
    # We use a Counter to count all tags
    all_tags = []
    # iterating is slower but memory safe for massive lists,
    # but with 220GB RAM we can probably do a flat map
    # Let's use a faster approach: str.split(expand=True) is too wide.
    # We'll join and split, but that makes a huge string.
    # Best approach for 4M rows: iterate in chunks or use Counter update
    tag_counts = Counter()

    # Processing in chunks to be safe and show progress logic internally
    chunk_size = 100000
    for i in range(0, len(tags_series), chunk_size):
        chunk = tags_series.iloc[i : i + chunk_size]
        # Split and flatten
        batch_tags = [tag for row in chunk for tag in row.split()]
        tag_counts.update(batch_tags)

    total_unique_tags = len(tag_counts)
    total_tag_occurrences = sum(tag_counts.values())

    print("\n--- Class Balance / Frequency ---")
    print(f"Total Unique Tags: {total_unique_tags}")
    print(f"Total Tag Occurrences: {total_tag_occurrences}")

    # Top 20 Tags
    print("\nTop 20 Most Frequent Tags:")
    most_common = tag_counts.most_common(20)
    for tag, count in most_common:
        freq = count / len(df)
        print(f"  {tag:<20}: {count} ({freq*100:.2f}% of questions)")

    # Rare tags
    counts = np.array(list(tag_counts.values()))
    single_occurrence = np.sum(counts == 1)
    rare_ratio = single_occurrence / total_unique_tags

    print(
        f"\nTags with only 1 occurrence: {single_occurrence} ({rare_ratio*100:.2f}% of unique tags)"
    )

    # Imbalance metrics
    max_count = counts.max()
    min_count = counts.min()
    print(f"Most frequent class count: {max_count}")
    print(f"Least frequent class count: {min_count}")
    print(f"Imbalance Ratio (Max/Min): {max_count/min_count:.4f}")

    return tags_per_row


def clean_html(raw_html):
    # Simple regex to strip HTML tags for text analysis
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, " ", raw_html)
    return cleantext


def analyze_text_modality(df):
    print_section("Input Data Analysis (Text Modality)")

    # For text analysis, 4M rows is heavy for detailed tokenization in a short script.
    # We will use a representative sample of 200,000 rows for the deep text stats
    # to ensure the script finishes well within the hour.
    sample_size = min(200000, len(df))
    print(f"Analyzing a random sample of {sample_size} rows for text statistics...")

    df_sample = df.sample(n=sample_size, random_state=42).copy()

    # 1. Length Analysis (Characters and Words)
    # We analyze Title and Body separately

    # Title Analysis
    df_sample["title_char_len"] = df_sample["Title"].astype(str).apply(len)
    df_sample["title_word_count"] = (
        df_sample["Title"].astype(str).apply(lambda x: len(x.split()))
    )

    # Body Analysis - We should strip HTML for accurate word counts,
    # but raw length is also useful. We'll do raw char length and stripped word count.
    df_sample["body_raw_char_len"] = df_sample["Body"].astype(str).apply(len)

    # Strip HTML for word count (computationally more expensive)
    # Using a simple regex approach
    clean_body = df_sample["Body"].astype(str).apply(clean_html)
    df_sample["body_word_count"] = clean_body.apply(lambda x: len(x.split()))

    def report_stats(name, series):
        print(f"\n--- {name} Distribution ---")
        print(f"Mean: {series.mean():.4f}")
        print(f"Std:  {series.std():.4f}")
        print(f"Min:  {series.min():.4f}")
        print(f"Max:  {series.max():.4f}")

        # Outliers (IQR method)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[series > upper_bound].count()
        print(
            f"Outliers (> {upper_bound:.2f}): {outliers} ({outliers/len(series)*100:.2f}%)"
        )

    report_stats("Title Character Length", df_sample["title_char_len"])
    report_stats("Title Word Count", df_sample["title_word_count"])
    report_stats("Body Raw Character Length", df_sample["body_raw_char_len"])
    report_stats("Body (Cleaned) Word Count", df_sample["body_word_count"])

    # 2. Vocabulary Analysis
    # We'll combine Title and Clean Body for vocab analysis
    print("\n--- Vocabulary Analysis (Sampled) ---")

    # Tokenizer: Simple lowercase and split by non-alphanumeric
    # This is a basic approximation suitable for EDA
    def get_tokens(text):
        return re.findall(r"\b\w+\b", text.lower())

    # We'll iterate and update a counter
    vocab_counter = Counter()

    # Process combined text
    # Using a smaller subsample for vocab if 200k is too slow, but 200k should be fine
    # Let's do it on the 200k sample
    all_text = df_sample["Title"].astype(str) + " " + clean_body

    for text in all_text:
        vocab_counter.update(get_tokens(text))

    vocab_size = len(vocab_counter)
    total_tokens = sum(vocab_counter.values())

    print(f"Unique Vocabulary Size: {vocab_size}")
    print(f"Total Tokens in Sample: {total_tokens}")

    # OOV Potential (Hapax Legomena)
    # Words appearing only once
    singletons = sum(1 for count in vocab_counter.values() if count == 1)
    print(
        f"Words appearing only once (OOV potential): {singletons} ({singletons/vocab_size*100:.4f}% of vocab)"
    )

    # Top words
    print("\nTop 20 Most Common Words:")
    for word, count in vocab_counter.most_common(20):
        print(f"  {word}: {count}")

    return df_sample


def analyze_relationships(df_sample, tags_per_row):
    print_section("Feature/Signal Relationships")

    # We use the sample dataframe which already has lengths calculated.
    # We need to map the 'tags_per_row' (calculated on full df) to this sample.
    df_sample["num_tags"] = tags_per_row.loc[df_sample.index]

    print("--- Meta-Feature Correlations ---")
    # Correlations between lengths and number of tags
    cols_to_corr = [
        "title_char_len",
        "title_word_count",
        "body_raw_char_len",
        "body_word_count",
        "num_tags",
    ]
    corr_matrix = df_sample[cols_to_corr].corr(method="pearson")

    print("Pearson Correlation Matrix:")
    print(corr_matrix.round(4))

    print("\nKey Observations:")
    print(
        f"Correlation between Body Word Count and Number of Tags: {corr_matrix.loc['body_word_count', 'num_tags']:.4f}"
    )
    print(
        f"Correlation between Title Word Count and Number of Tags: {corr_matrix.loc['title_word_count', 'num_tags']:.4f}"
    )

    # Relationship between specific tags and length
    # Let's pick the top 3 tags from the global analysis (hardcoded based on typical StackOverflow data or previous step)
    # We'll re-detect top tags from the sample for safety
    sample_tags = df_sample["Tags"].astype(str).str.split(expand=True).stack()
    top_tags = sample_tags.value_counts().head(3).index.tolist()

    print(f"\n--- Relationship: Content Length vs Top Tags {top_tags} ---")

    for tag in top_tags:
        # Create binary indicator
        has_tag = df_sample["Tags"].astype(str).apply(lambda x: tag in x.split())
        avg_len = df_sample.loc[has_tag, "body_word_count"].mean()
        overall_avg = df_sample["body_word_count"].mean()

        print(f"Tag '{tag}':")
        print(f"  Avg Body Word Count: {avg_len:.4f} (Overall Avg: {overall_avg:.4f})")
        print(f"  Difference: {avg_len - overall_avg:.4f}")


def main():
    set_seed(42)

    # Define path
    train_path = "./metadata/train.csv"

    print(f"Loading dataset from {train_path}...")
    # Reading only necessary columns to save memory if needed, but we need all for EDA
    try:
        df = pd.read_csv(train_path)
        print(f"Dataset loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: metadata/train.csv not found.")
        return

    # 1. Target Variable Analysis
    tags_per_row = analyze_targets(df)

    # 2. Input Data Analysis (Text)
    # Returns a sample dataframe with computed length features
    df_sample = analyze_text_modality(df)

    # 3. Feature Relationships
    analyze_relationships(df_sample, tags_per_row)

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
