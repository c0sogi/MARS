import os
import json
import random
import numpy as np
import pandas as pd
import warnings
from collections import Counter
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SAMPLE_SIZE = 2000  # Number of notebooks to sample for EDA
RANDOM_STATE = 42

# Suppress warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def main():
    # --------------------------------------------------------------------------
    # 1. Data Loading & Sampling
    # --------------------------------------------------------------------------
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_meta = pd.read_csv(METADATA_PATH)

    # Sample notebooks to keep runtime low
    if len(df_meta) > SAMPLE_SIZE:
        df_sample = df_meta.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
    else:
        df_sample = df_meta.copy()

    # --------------------------------------------------------------------------
    # 2. Data Processing (Parsing JSONs)
    # --------------------------------------------------------------------------
    cell_data = []
    notebook_data = []

    # Vocab counters
    code_vocab = Counter()
    md_vocab = Counter()

    # We will limit vocab processing to avoid OOM or excessive time on tokenization
    # Process max 100k cells for vocab
    vocab_cell_limit = 100000
    processed_vocab_cells = 0

    for _, row in df_sample.iterrows():
        nb_id = row["id"]
        rel_path = row["filepath"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Ground truth order
        order_str = row["cell_order"]
        if pd.isna(order_str):
            continue
        correct_order = order_str.split()
        rank_map = {cell_id: i for i, cell_id in enumerate(correct_order)}
        total_cells = len(correct_order)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                nb_json = json.load(f)
        except Exception:
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        num_code = 0
        num_md = 0

        for cell_id in correct_order:
            c_type = cell_types.get(cell_id, "unknown")
            c_source = sources.get(cell_id, "")

            # Basic stats
            char_len = len(c_source)
            # Simple whitespace split for word count
            words = c_source.split()
            word_len = len(words)

            rank = rank_map.get(cell_id, -1)
            norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

            is_markdown = 1 if c_type == "markdown" else 0

            if is_markdown:
                num_md += 1
            else:
                num_code += 1

            # Store cell data
            cell_data.append(
                {
                    "notebook_id": nb_id,
                    "cell_id": cell_id,
                    "is_markdown": is_markdown,
                    "char_len": char_len,
                    "word_len": word_len,
                    "rank": rank,
                    "norm_rank": norm_rank,
                    "total_cells_in_nb": total_cells,
                }
            )

            # Vocab accumulation (partial)
            if processed_vocab_cells < vocab_cell_limit:
                # Limit tokens per cell to avoid huge processing on massive code dumps
                tokens = words[:500]
                if is_markdown:
                    md_vocab.update(tokens)
                else:
                    code_vocab.update(tokens)
                processed_vocab_cells += 1

        notebook_data.append(
            {
                "notebook_id": nb_id,
                "total_cells": total_cells,
                "num_code": num_code,
                "num_md": num_md,
                "md_ratio": num_md / total_cells if total_cells > 0 else 0,
            }
        )

    df_cells = pd.DataFrame(cell_data)
    df_notebooks = pd.DataFrame(notebook_data)

    # --------------------------------------------------------------------------
    # 3. Target Variable Analysis
    # --------------------------------------------------------------------------
    # In this ranking task, the "target" behavior is best described by the
    # position (Normalized Rank) of the cells.

    print("DATA INTEGRITY")
    print(f"Analysis performed on {len(df_sample)} training notebooks.")
    print(f"Total cells processed: {len(df_cells)}")
    print("-" * 30)

    print("\nTARGET VARIABLE ANALYSIS")
    # Distribution of Normalized Rank for All Cells (should be uniform by definition of rank)
    # Distribution of Normalized Rank for Markdown Cells (This is the key insight)
    md_ranks = df_cells[df_cells["is_markdown"] == 1]["norm_rank"]

    print("Target: Normalized Rank of Markdown Cells (0.0=Top, 1.0=Bottom)")
    print(f"Mean Rank: {md_ranks.mean():.4f}")
    print(f"Std Dev:   {md_ranks.std():.4f}")

    # Skewness/Kurtosis
    skew_val = skew(md_ranks)
    kurt_val = kurtosis(md_ranks)
    print(f"Skewness:  {skew_val:.4f} (Pos: tail right, Neg: tail left)")
    print(f"Kurtosis:  {kurt_val:.4f}")

    # Class Balance
    total_code = df_notebooks["num_code"].sum()
    total_md = df_notebooks["num_md"].sum()
    total_all = total_code + total_md
    print(f"Class Balance (Cell Type):")
    print(f"  Code Cells:     {total_code} ({total_code/total_all*100:.2f}%)")
    print(f"  Markdown Cells: {total_md} ({total_md/total_all*100:.2f}%)")

    # --------------------------------------------------------------------------
    # 4. Input Data Analysis (Text Modality)
    # --------------------------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # Length Analysis
    def get_stats(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
        return series.mean(), series.std(), series.min(), series.max(), outliers

    print("Sequence Lengths (Character Counts):")
    for ctype, name in [(0, "Code"), (1, "Markdown")]:
        subset = df_cells[df_cells["is_markdown"] == ctype]["char_len"]
        mean_v, std_v, min_v, max_v, out_v = get_stats(subset)
        print(
            f"  {name}: Mean={mean_v:.4f}, Std={std_v:.4f}, Min={min_v}, Max={max_v}, Outliers={out_v}"
        )

    print("Sequence Lengths (Word Counts):")
    for ctype, name in [(0, "Code"), (1, "Markdown")]:
        subset = df_cells[df_cells["is_markdown"] == ctype]["word_len"]
        mean_v, std_v, min_v, max_v, out_v = get_stats(subset)
        print(
            f"  {name}: Mean={mean_v:.4f}, Std={std_v:.4f}, Min={min_v}, Max={max_v}, Outliers={out_v}"
        )

    # Vocabulary Analysis
    print("Vocabulary Statistics (Estimated from sample):")
    print(f"  Code Vocab Size:     {len(code_vocab)}")
    print(f"  Markdown Vocab Size: {len(md_vocab)}")
    # Overlap
    common_tokens = set(code_vocab.keys()) & set(md_vocab.keys())
    print(f"  Shared Vocabulary:   {len(common_tokens)}")

    # --------------------------------------------------------------------------
    # 5. Feature/Signal Relationships
    # --------------------------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Structured Relationships (Correlation)
    # We correlate features with 'norm_rank'
    # We use Spearman because rank is ordinal/monotonic
    corr_cols = ["char_len", "word_len", "is_markdown", "norm_rank"]
    corr_matrix = df_cells[corr_cols].corr(method="spearman")

    print("Spearman Correlation with Normalized Rank:")
    print(f"  vs Char Length:  {corr_matrix.loc['char_len', 'norm_rank']:.4f}")
    print(f"  vs Word Length:  {corr_matrix.loc['word_len', 'norm_rank']:.4f}")
    print(f"  vs Is_Markdown:  {corr_matrix.loc['is_markdown', 'norm_rank']:.4f}")

    # Feature Importance (Random Forest)
    # Task: Predict normalized rank based on cell features
    print("Feature Importance (RF Regressor predicting Rank):")
    X = df_cells[["char_len", "word_len", "is_markdown"]]
    y = df_cells["norm_rank"]

    # Subsample for training RF to be fast
    if len(X) > 10000:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=10000, random_state=RANDOM_STATE
        )
    else:
        X_sub, y_sub = X, y

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_sub, y_sub)

    importances = rf.feature_importances_
    feat_names = X.columns
    indices = np.argsort(importances)[::-1]

    for f in range(len(feat_names)):
        print(f"  {f+1}. {feat_names[indices[f]]}: {importances[indices[f]]:.4f}")

    # Unstructured / Metadata Relationships
    # Relationship between Notebook Length (total cells) and Markdown Ratio
    nb_corr = df_notebooks["total_cells"].corr(df_notebooks["md_ratio"])
    print("Meta-Feature Relationship:")
    print(f"  Correlation (Total Cells vs Markdown Ratio): {nb_corr:.4f}")
    print(
        f"  (Interpretation: Do longer notebooks have proportionally more documentation?)"
    )


if __name__ == "__main__":
    main()
