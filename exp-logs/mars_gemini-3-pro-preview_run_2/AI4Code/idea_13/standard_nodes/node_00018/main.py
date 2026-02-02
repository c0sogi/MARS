import os
import sys
import gc
import warnings
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings("ignore")

# Monkeypatch tqdm to ensure silent execution as per requirements
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# Import provided libraries
from library.config import Config
from library.utils import seed_everything, kendall_tau_metric
from library.stage1_ridge import Stage1Ridge
from library.stage2_metric import Stage2Metric
from library.stage3_lgbm import Stage3LGBM
from library.data_loader import load_metadata


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Adjust configuration for a fast baseline execution within 2 hours
    # We reduce epochs and estimators to ensure completion while using the full dataset.
    Config.METRIC_EPOCHS = 2
    Config.LGBM_PARAMS["n_estimators"] = 800
    Config.LGBM_PARAMS["early_stopping_rounds"] = 50
    Config.LGBM_PARAMS["verbose"] = -1

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Stage 1: Sparse Lexical Regression (Ridge)
    # --------------------------------------------------------------------------
    # Generates OOF predictions using sparse TF-IDF features.
    # These predictions act as a "Signpost" feature for the final stack.
    stage1 = Stage1Ridge(Config)
    stage1.run(load_cached_preds=True)

    # Free memory
    del stage1
    gc.collect()

    # --------------------------------------------------------------------------
    # 3. Stage 2: Supervised Metric Learning
    # --------------------------------------------------------------------------
    # Trains a Siamese network to align Markdown and Code embeddings in a shared space.
    # This enables finding semantically relevant code anchors even with disjoint vocabulary.
    stage2 = Stage2Metric(Config)
    stage2.train()

    # Free memory (especially GPU)
    del stage2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # 4. Stage 3: Neighborhood Gradient Booster (LightGBM)
    # --------------------------------------------------------------------------
    # Generates anchor features (Top-K nearest code cells) and trains the final ranker.
    stage3 = Stage3LGBM(Config)
    model = stage3.train(load_cached_features=True)

    # --------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    # Load validation features (generated and saved during Stage 3 training)
    if not os.path.exists(Config.VAL_FEATURES_PATH):
        raise FileNotFoundError(
            f"Validation features not found at {Config.VAL_FEATURES_PATH}"
        )

    df_val = pd.read_parquet(Config.VAL_FEATURES_PATH)

    # Identify Markdown cells (the targets of our prediction)
    val_mask = df_val["cell_type"] == "markdown"

    # Define features used by the model
    feature_cols = [
        "ridge_pred",
        "anchor_mean_rank",
        "anchor_weighted_rank",
        "anchor_min_dist",
        "anchor_nearest_rank",
    ]

    # Predict on validation set
    X_val = df_val.loc[val_mask, feature_cols]
    val_preds = model.predict(X_val)
    df_val.loc[val_mask, "pred_rank"] = val_preds

    # Assign ranks to code cells (linear interpolation 0..1)
    # Code cells are fixed "anchors" in the sequence.
    def assign_code_ranks(group):
        code_mask = group["cell_type"] == "code"
        n_code = code_mask.sum()
        if n_code > 0:
            ranks = np.linspace(0.0, 1.0, n_code)
            group.loc[code_mask, "pred_rank"] = ranks
        return group

    df_val = df_val.groupby("id", group_keys=False).apply(assign_code_ranks)
    df_val["pred_rank"] = df_val["pred_rank"].fillna(0.0)

    # Construct predicted ordering by sorting cells by their predicted rank
    df_val_sorted = df_val.sort_values(["id", "pred_rank"])
    pred_orders = (
        df_val_sorted.groupby("id")["cell_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    pred_orders.columns = ["id", "cell_order"]

    # Load Ground Truth
    val_meta = load_metadata("val")

    # Calculate Metric
    final_metric = kendall_tau_metric(pred_orders, val_meta)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error for markdown cells
    df_analysis = df_val[val_mask].copy()
    df_analysis["error"] = (df_analysis["pred_rank"] - df_analysis["norm_rank"]).abs()

    # Add auxiliary features for correlation analysis
    df_analysis["source_len"] = df_analysis["source"].fillna("").str.len()

    analysis_cols = feature_cols + ["n_cells", "source_len"]
    correlations = {}

    for col in analysis_cols:
        if col in df_analysis.columns:
            corr = df_analysis["error"].corr(df_analysis[col])
            correlations[col] = corr

    print("Failure Analysis (Correlation with Error):")
    for col, corr in correlations.items():
        print(f"{col}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7959051868218839

    if final_metric > THRESHOLD:
        stage3.predict(load_cached_features=True)
    else:
        print(
            f"Metric {final_metric} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
