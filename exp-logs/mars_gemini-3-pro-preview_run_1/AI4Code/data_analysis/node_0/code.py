import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_data():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train_metadata.csv"

    print("Loading metadata...")
    df_meta = pd.read_csv(METADATA_PATH)

    # Sample data to keep runtime efficient
    SAMPLE_SIZE = 5000
    if len(df_meta) > SAMPLE_SIZE:
        df_sample = df_meta.sample(n=SAMPLE_SIZE, random_state=42).reset_index(
            drop=True
        )
    else:
        df_sample = df_meta.copy()

    print(f"Analyzing {len(df_sample)} notebooks...")

    # Containers for analysis
    notebook_stats = []
    cell_data = []
    vocab_counter = Counter()

    # Iterate through sampled notebooks
    for idx, row in df_sample.iterrows():
        nb_id = row["id"]
        filepath = os.path.join(INPUT_DIR, row["filepath"])
        ground_truth_order = row["cell_order"].split()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                nb_json = json.load(f)
        except Exception:
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        # Determine positions
        # Create a map from cell_id to rank (0 to N-1)
        rank_map = {cell_id: r for r, cell_id in enumerate(ground_truth_order)}

        code_cells = [cid for cid, ctype in cell_types.items() if ctype == "code"]
        md_cells = [cid for cid, ctype in cell_types.items() if ctype == "markdown"]

        # Notebook Level Stats
        nb_stat = {
            "id": nb_id,
            "num_code": len(code_cells),
            "num_md": len(md_cells),
            "total_cells": len(ground_truth_order),
        }
        notebook_stats.append(nb_stat)

        # Cell Level Stats (Focusing on Markdown for "Target" analysis relative to Code)
        # We also collect Code stats for comparison

        total_cells = len(ground_truth_order)

        for cell_id in ground_truth_order:
            if cell_id not in sources:
                continue

            ctype = cell_types.get(cell_id, "unknown")
            text = sources[cell_id]

            # Text Stats
            char_len = len(text)
            words = text.split()
            word_count = len(words)

            # Update Vocab (lightweight)
            if idx < 100:  # Only update vocab for first 100 nbs to save time/memory
                vocab_counter.update(words)

            # Target: Relative Rank (0.0 to 1.0)
            rank = rank_map[cell_id]
            rel_rank = rank / (total_cells - 1) if total_cells > 1 else 0.5

            c_data = {
                "cell_type": ctype,
                "char_len": char_len,
                "word_count": word_count,
                "rel_rank": rel_rank,
                "is_header": 1 if text.strip().startswith("#") else 0,
                "num_code_context": len(code_cells),
                "num_md_context": len(md_cells),
            }
            cell_data.append(c_data)

    df_nb = pd.DataFrame(notebook_stats)
    df_cells = pd.DataFrame(cell_data)

    # Separate Markdown and Code
    df_md = df_cells[df_cells["cell_type"] == "markdown"].copy()
    df_code = df_cells[df_cells["cell_type"] == "code"].copy()

    # ==========================================
    # 2. TARGET VARIABLE ANALYSIS
    # ==========================================
    print("\n2. TARGET VARIABLE ANALYSIS")

    # In this ranking task, the "Target" for a markdown cell is its Relative Rank.
    # We analyze the distribution of this target.

    target_mean = df_md["rel_rank"].mean()
    target_std = df_md["rel_rank"].std()
    target_skew = skew(df_md["rel_rank"])
    target_kurt = kurtosis(df_md["rel_rank"])

    print(f"Target Variable: Markdown Cell Relative Rank (0.0=Top, 1.0=Bottom)")
    print(f"Mean: {target_mean:.4f}")
    print(f"Std Dev: {target_std:.4f}")
    print(f"Skewness: {target_skew:.4f}")
    print(f"Kurtosis: {target_kurt:.4f}")

    # Notebook Composition (Class Balance equivalent)
    avg_code = df_nb["num_code"].mean()
    avg_md = df_nb["num_md"].mean()
    ratio = avg_md / avg_code if avg_code > 0 else 0

    print(f"Notebook Composition:")
    print(f"Avg Code Cells: {avg_code:.4f}")
    print(f"Avg Markdown Cells: {avg_md:.4f}")
    print(f"Markdown/Code Ratio: {ratio:.4f}")

    # ==========================================
    # 3. INPUT DATA ANALYSIS (TEXT)
    # ==========================================
    print("\n3. INPUT DATA ANALYSIS (TEXT)")

    # Lengths
    print("Sequence Lengths (Character Counts):")
    print(
        f"Code - Mean: {df_code['char_len'].mean():.4f}, Max: {df_code['char_len'].max()}"
    )
    print(
        f"Markdown - Mean: {df_md['char_len'].mean():.4f}, Max: {df_md['char_len'].max()}"
    )

    print("Sequence Lengths (Word Counts):")
    print(f"Code - Mean: {df_code['word_count'].mean():.4f}")
    print(f"Markdown - Mean: {df_md['word_count'].mean():.4f}")

    # Vocabulary
    # Note: This is an approximation from the subsample
    print("Vocabulary Statistics (Sampled):")
    print(f"Unique Tokens (approx): {len(vocab_counter)}")
    common_tokens = vocab_counter.most_common(5)
    print(f"Most Common Tokens: {common_tokens}")

    # Missing/Empty
    empty_md = df_md[df_md["char_len"] == 0].shape[0]
    print(f"Empty Markdown Cells: {empty_md} ({empty_md/len(df_md)*100:.2f}%)")

    # ==========================================
    # 4. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("\n4. FEATURE/SIGNAL RELATIONSHIPS")

    # Structured Relationships (Correlation)
    # We focus on Markdown cells: Do features correlate with Relative Rank?
    corr_cols = [
        "char_len",
        "word_count",
        "is_header",
        "num_code_context",
        "num_md_context",
        "rel_rank",
    ]
    corr_matrix = df_md[corr_cols].corr(method="spearman")

    print("Spearman Correlation with Target (Relative Rank):")
    print(corr_matrix["rel_rank"].sort_values(ascending=False))

    # Redundancy
    print("Redundancy Check (Correlation > 0.90):")
    high_corr_pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            c1, c2 = corr_cols[i], corr_cols[j]
            val = corr_matrix.loc[c1, c2]
            if abs(val) > 0.90:
                high_corr_pairs.append((c1, c2, val))

    if high_corr_pairs:
        for c1, c2, val in high_corr_pairs:
            print(f"{c1} - {c2}: {val:.4f}")
    else:
        print("No highly collinear pairs found.")

    # Feature Importance (Random Forest)
    # Predict Relative Rank based on cell features
    print("Feature Importance (Random Forest Regressor):")

    # Prepare data for RF
    # Downsample cells for RF training to save time if needed, but 5000 nbs * ~15 md cells = 75k rows, which is fine.
    X = df_md[
        ["char_len", "word_count", "is_header", "num_code_context", "num_md_context"]
    ].fillna(0)
    y = df_md["rel_rank"]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print(importances.head(5))

    # Meta-Feature Relationship
    # Does the number of code cells affect the position of markdown cells?
    # We can infer this from the correlation matrix above (num_code_context vs rel_rank)
    # Let's explicitly check if longer notebooks have markdown cells pushed to the bottom.

    print("Meta-Feature Insight:")
    corr_code_rank = df_md["num_code_context"].corr(df_md["rel_rank"])
    print(
        f"Correlation between Num Code Cells and Markdown Relative Rank: {corr_code_rank:.4f}"
    )
    if corr_code_rank > 0.1:
        print(
            "-> Larger notebooks tend to have markdown cells distributed lower (later) in the order."
        )
    elif corr_code_rank < -0.1:
        print(
            "-> Larger notebooks tend to have markdown cells distributed higher (earlier) in the order."
        )
    else:
        print(
            "-> No strong linear relationship between notebook size and markdown distribution."
        )


if __name__ == "__main__":
    analyze_data()
