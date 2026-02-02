import os
import json
import pandas as pd
import numpy as np
import random
from scipy.stats import skew, kurtosis
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def analyze_notebooks():
    # 1. Setup and Data Loading
    set_seed(42)
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    print("DATA INTEGRITY CHECK")
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # Sample 10,000 notebooks for efficient EDA
    SAMPLE_SIZE = 10000
    if len(df_train) > SAMPLE_SIZE:
        df_sampled = df_train.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        df_sampled = df_train.copy()

    print(
        f"Analyzing {len(df_sampled)} notebooks sampled from {len(df_train)} training records."
    )

    # Storage for aggregated statistics
    notebook_stats = []
    all_code_lens = []
    all_md_lens = []
    vocab_counter = Counter()

    # 2. Processing Notebooks (Input Data Analysis)
    print("\nProcessing sampled notebooks to extract text statistics...")

    for _, row in df_sampled.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        n_code = 0
        n_md = 0
        curr_code_lens = []
        curr_md_lens = []

        # Iterate through cells in the notebook
        for cell_id, c_type in cell_types.items():
            source_text = sources.get(cell_id, "")
            length = len(source_text)

            # Basic whitespace tokenization for vocab estimation
            # Limit to first 500 chars per cell to optimize runtime
            tokens = source_text[:500].split()

            if c_type == "code":
                n_code += 1
                curr_code_lens.append(length)
                vocab_counter.update(tokens)
            elif c_type == "markdown":
                n_md += 1
                curr_md_lens.append(length)
                vocab_counter.update(tokens)

        # Record notebook-level metrics
        notebook_stats.append(
            {
                "n_code": n_code,
                "n_md": n_md,
                "n_total": n_code + n_md,
                "mean_code_len": np.mean(curr_code_lens) if curr_code_lens else 0,
                "mean_md_len": np.mean(curr_md_lens) if curr_md_lens else 0,
            }
        )

        all_code_lens.extend(curr_code_lens)
        all_md_lens.extend(curr_md_lens)

    df_stats = pd.DataFrame(notebook_stats)

    # 3. Target Variable Analysis
    # In this ranking task, the "Target" is the order. The complexity is defined by the number of items to sort (Markdown cells).
    print("\nTARGET VARIABLE ANALYSIS")
    print(
        "Note: The 'Target' in this ranking task is the cell order. We analyze the distribution of Markdown cells (items to sort) as a proxy for task complexity."
    )

    md_counts = df_stats["n_md"]
    print("\nDistribution of Markdown Cells (Sorting Targets):")
    print(f"Mean: {md_counts.mean():.4f}")
    print(f"Std:  {md_counts.std():.4f}")
    print(f"Min:  {md_counts.min():.4f}")
    print(f"Max:  {md_counts.max():.4f}")
    print(f"Skewness: {skew(md_counts):.4f}")
    print(f"Kurtosis: {kurtosis(md_counts):.4f}")

    code_counts = df_stats["n_code"]
    print("\nDistribution of Code Cells (Context Anchors):")
    print(f"Mean: {code_counts.mean():.4f}")
    print(f"Std:  {code_counts.std():.4f}")

    # 4. Input Data Analysis (Text Modality)
    print("\nINPUT DATA ANALYSIS (TEXT)")

    code_len_arr = np.array(all_code_lens)
    md_len_arr = np.array(all_md_lens)

    print("\nCode Cell Character Lengths:")
    if len(code_len_arr) > 0:
        print(f"Mean: {np.mean(code_len_arr):.4f}")
        print(f"Std:  {np.std(code_len_arr):.4f}")
        print(f"Max:  {np.max(code_len_arr):.4f}")
        # Outlier detection (IQR method)
        q75, q25 = np.percentile(code_len_arr, [75, 25])
        iqr = q75 - q25
        outlier_thresh = q75 + 1.5 * iqr
        outliers = np.sum(code_len_arr > outlier_thresh)
        print(
            f"Outliers (> {outlier_thresh:.2f} chars): {outliers} ({outliers/len(code_len_arr)*100:.2f}%)"
        )
    else:
        print("No code cells found.")

    print("\nMarkdown Cell Character Lengths:")
    if len(md_len_arr) > 0:
        print(f"Mean: {np.mean(md_len_arr):.4f}")
        print(f"Std:  {np.std(md_len_arr):.4f}")
        print(f"Max:  {np.max(md_len_arr):.4f}")
        # Outlier detection
        q75, q25 = np.percentile(md_len_arr, [75, 25])
        iqr = q75 - q25
        outlier_thresh = q75 + 1.5 * iqr
        outliers = np.sum(md_len_arr > outlier_thresh)
        print(
            f"Outliers (> {outlier_thresh:.2f} chars): {outliers} ({outliers/len(md_len_arr)*100:.2f}%)"
        )
    else:
        print("No markdown cells found.")

    print("\nVocabulary Statistics (Estimated):")
    print(f"Unique Tokens Found: {len(vocab_counter)}")
    print(f"Top 5 Tokens: {[t[0] for t in vocab_counter.most_common(5)]}")

    # 5. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Correlation: Code Count vs Markdown Count
    corr_counts = df_stats["n_code"].corr(df_stats["n_md"])
    print(f"Correlation (Num Code Cells vs Num Markdown Cells): {corr_counts:.4f}")

    # Correlation: Notebook Size vs Average Markdown Length
    # Do larger notebooks have more verbose explanations?
    corr_size_len = df_stats["n_total"].corr(df_stats["mean_md_len"])
    print(f"Correlation (Total Cells vs Mean Markdown Length): {corr_size_len:.4f}")

    # Correlation: Code Length vs Markdown Length
    # Do users who write long code also write long descriptions?
    corr_lens = df_stats["mean_code_len"].corr(df_stats["mean_md_len"])
    print(f"Correlation (Mean Code Length vs Mean Markdown Length): {corr_lens:.4f}")


if __name__ == "__main__":
    analyze_notebooks()
