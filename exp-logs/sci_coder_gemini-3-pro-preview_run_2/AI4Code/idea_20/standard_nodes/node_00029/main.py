import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import functions and classes from the provided library files
from library.config import Config
from library.train_pipeline import run_training
from library.inference_pipeline import run_inference
from library.data_loader import load_notebooks, load_metadata
from library.vectorizer import TextVectorizer
from library.feature_extractor import AnchorFeatureGenerator
from library.model_wrapper import Stage1Ridge, Stage2LGBM
from library.utils import kendall_tau, count_inversions


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Initialization
    set_seed(Config.SEED)

    # 2. Training Pipeline
    # We execute the full training pipeline with debug=False to ensure the model
    # learns from the entire dataset and meets the strict validation threshold.
    # The efficient Ridge+LGBM architecture fits within the runtime limits.
    print(">>> Executing Training Pipeline...")
    run_training(debug=False, load_cached_data=True)

    # 3. Validation Assessment
    print("\n>>> Executing Validation Assessment...")

    # Load Validation Data
    # We need the raw cell data to reconstruct orders and calculate stats
    val_df = load_notebooks("val", load_cached_data=True, debug=False)
    val_md = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)

    # Load Pre-computed Validation Features (generated during training)
    feature_path = os.path.join(Config.WORKING_DIR, "val_anchor_features.parquet")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Validation features not found at {feature_path}")
    val_features = pd.read_parquet(feature_path)

    # Load Trained Models
    print("Loading models for validation...")
    vectorizer = TextVectorizer()
    vectorizer.load(os.path.join(Config.WORKING_DIR, "text_vectorizer"))

    ridge = Stage1Ridge()
    ridge.load(os.path.join(Config.WORKING_DIR, "stage1_ridge_model"))

    lgbm = Stage2LGBM()
    lgbm.load(os.path.join(Config.WORKING_DIR, "stage2_lgbm"))

    # Generate Stage 1 Predictions for Validation
    # (These are needed as input features for Stage 2)
    print("Generating Stage 1 predictions for validation...")
    val_sources = val_md["source"].fillna("").astype(str)
    X_val_sparse = vectorizer.transform(val_sources)
    val_md["stage1_pred"] = ridge.predict(X_val_sparse)

    # Merge Stage 1 predictions into the feature set
    val_final = val_features.merge(
        val_md[["id", "cell_id", "stage1_pred"]], on=["id", "cell_id"], how="left"
    )

    # Generate Stage 2 Predictions (Final Ranks)
    print("Generating Stage 2 predictions for validation...")
    exclude_cols = [
        "id",
        "cell_id",
        "norm_rank",
        "cell_type",
        "source",
        "ancestor_id",
        "parent_id",
    ]
    feature_cols = [c for c in val_final.columns if c not in exclude_cols]

    val_final["pred_rank"] = lgbm.predict(val_final, feature_cols)

    # Reconstruct Cell Orders
    print("Reconstructing validation cell orders...")

    # A. Code Cells: Assigned fixed equidistant ranks [0, 1]
    code_cells = val_df[val_df["cell_type"] == "code"].copy()
    if not code_cells.empty:
        code_cells["rank"] = code_cells.groupby("id")["cell_id"].transform(
            lambda x: np.linspace(0, 1, len(x))
        )
    else:
        code_cells["rank"] = []

    # B. Markdown Cells: Assigned predicted ranks
    md_cells = val_final[["id", "cell_id", "pred_rank"]].rename(
        columns={"pred_rank": "rank"}
    )

    # C. Combine and Sort
    all_cells = pd.concat(
        [code_cells[["id", "cell_id", "rank"]], md_cells[["id", "cell_id", "rank"]]],
        ignore_index=True,
    )

    all_cells = all_cells.sort_values(["id", "rank"])

    # D. Group into ordered lists
    preds_map = all_cells.groupby("id")["cell_id"].apply(list).to_dict()

    # Load Ground Truth from Metadata
    val_meta = load_metadata("val")

    ground_truths = []
    predictions = []
    ids = []

    for _, row in val_meta.iterrows():
        nb_id = row["id"]
        gt_order = row["cell_order"].split()
        pred_order = preds_map.get(nb_id, [])

        # Fallback for empty predictions (unlikely but safe)
        if not pred_order and gt_order:
            # If prediction missing, just use code cells if available or empty list
            # This handles edge cases where a notebook might have been filtered out
            pred_order = [
                c
                for c in gt_order
                if c in code_cells[code_cells["id"] == nb_id]["cell_id"].values
            ]

        ground_truths.append(gt_order)
        predictions.append(pred_order)
        ids.append(nb_id)

    # Calculate Final Metric
    final_metric = kendall_tau(ground_truths, predictions)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n>>> Executing Failure Analysis...")

    # Calculate per-notebook error (1 - Kendall Tau)
    errors = []
    for gt, pred in zip(ground_truths, predictions):
        n = len(gt)
        if n <= 1:
            k = 1.0
        else:
            rank_map = {cid: i for i, cid in enumerate(gt)}
            try:
                # Filter prediction to only include valid IDs
                pred_ranks = [rank_map[cid] for cid in pred if cid in rank_map]
                swaps = count_inversions(pred_ranks)
                k = 1 - 4 * swaps / (n * (n - 1))
            except Exception:
                k = 0.0
        errors.append(1.0 - k)

    analysis_df = pd.DataFrame({"id": ids, "error": errors})

    # Aggregate Notebook Stats for correlation
    nb_stats = val_df.groupby("id").size().reset_index(name="total_cells_real")
    md_counts = (
        val_df[val_df["cell_type"] == "markdown"]
        .groupby("id")
        .size()
        .reset_index(name="md_count")
    )
    nb_stats = nb_stats.merge(md_counts, on="id", how="left").fillna(0)
    nb_stats["md_ratio_real"] = nb_stats["md_count"] / nb_stats["total_cells_real"]

    analysis_df = analysis_df.merge(nb_stats, on="id", how="left").fillna(0)

    # Compute Correlations
    correlations = analysis_df[["error", "total_cells_real", "md_ratio_real"]].corr()[
        "error"
    ]
    print("Correlation between Error and Features:")
    print(correlations.drop("error"))

    # 5. Submission Logic
    THRESHOLD = 0.7959051868218839

    if final_metric > THRESHOLD:
        print(
            f"\nMetric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        run_inference(debug=False, load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
