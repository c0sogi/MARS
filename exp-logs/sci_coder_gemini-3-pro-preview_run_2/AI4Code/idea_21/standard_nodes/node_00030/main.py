import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
import torch
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import kendall_tau_metric, convert_ranks_to_order
from library.data_manager import DataManager
from library.vectorizer import DualVectorizer
from library.anchor_engine import AnchorExtractor
from library.stage1_ridge import RidgeStacker
from library.stage2_lgbm import LGBMRanker

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def add_metadata_features(df_features, df_cells):
    """
    Computes and merges notebook-level metadata features:
    - n_cells: Total number of cells
    - md_ratio: Ratio of markdown cells
    """
    # Calculate metadata per notebook
    nb_stats = (
        df_cells.groupby("id")
        .apply(
            lambda x: pd.Series(
                {"n_cells": len(x), "md_ratio": (x["cell_type"] == "markdown").mean()}
            )
        )
        .reset_index()
    )

    # Merge with features
    # df_features has 'cell_id', we need to map via 'id'
    # But df_features might not have 'id' if it came purely from AnchorExtractor
    # AnchorExtractor output usually has 'cell_id' but not 'id' explicitly in columns unless added?
    # Let's check AnchorExtractor code: it creates dicts with 'cell_id'.
    # We need to map cell_id to notebook id.

    cell_to_nb = df_cells[["cell_id", "id"]].drop_duplicates()

    df_merged = df_features.merge(cell_to_nb, on="cell_id", how="left")
    df_merged = df_merged.merge(nb_stats, on="id", how="left")

    # Drop 'id' as it's not a feature for the model
    # Keep 'cell_id' for tracking if needed, but drop before training
    return df_merged


def reconstruct_orders(df_cells, df_preds):
    """
    Reconstructs cell orders for a set of notebooks based on predictions.
    df_cells: DataFrame containing all cells (Code + Markdown)
    df_preds: DataFrame containing ['cell_id', 'pred_rank'] for markdown cells
    """
    # Create a dict of preds
    pred_map = dict(zip(df_preds["cell_id"], df_preds["pred_rank"]))

    results = []

    for nb_id, group in df_cells.groupby("id"):
        code_cells = group[group["cell_type"] != "markdown"]["cell_id"].tolist()
        md_cells = group[group["cell_type"] == "markdown"]["cell_id"].tolist()

        # Get ranks for markdown cells
        md_ranks = [pred_map.get(cid, 0.5) for cid in md_cells]

        # Convert to order string
        order_str = convert_ranks_to_order(code_cells, md_cells, md_ranks)
        results.append({"id": nb_id, "cell_order": order_str})

    return pd.DataFrame(results)


def main():
    # --------------------------------------------------------------------------
    # 0. Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.RANDOM_STATE)

    # --------------------------------------------------------------------------
    # 1. Data Loading
    # --------------------------------------------------------------------------
    dm = DataManager()

    # Load Validation Data (Full)
    df_val = dm.load_data("val", load_cached_data=True)

    # Load Training Data
    df_train_full = dm.load_data("train", load_cached_data=True)

    # Subsample Training Data for Fast Baseline
    # We group by ancestor_id to ensure we don't leak groups if we were doing CV,
    # but mainly to keep the logic consistent.
    SAMPLE_ANCESTORS = 10000
    unique_ancestors = df_train_full["ancestor_id"].unique()

    if len(unique_ancestors) > SAMPLE_ANCESTORS:
        selected_ancestors = np.random.choice(
            unique_ancestors, SAMPLE_ANCESTORS, replace=False
        )
        df_train = df_train_full[
            df_train_full["ancestor_id"].isin(selected_ancestors)
        ].copy()
    else:
        df_train = df_train_full.copy()

    # --------------------------------------------------------------------------
    # 2. Vectorization
    # --------------------------------------------------------------------------
    vec = DualVectorizer()

    # Fit on Markdown sources from the sampled training set
    train_md_sources = (
        df_train[df_train["cell_type"] == "markdown"]["source"].astype(str).tolist()
    )
    vec.fit_or_load(train_md_sources, load_cached_data=True)

    # Transform All Sources (Code + Markdown)
    # We handle NaNs by converting to empty string just in case
    tfidf_train, svd_train = vec.transform(
        df_train["source"].fillna("").astype(str).tolist()
    )
    tfidf_val, svd_val = vec.transform(df_val["source"].fillna("").astype(str).tolist())

    # --------------------------------------------------------------------------
    # 3. Stage 1: Ridge Regression
    # --------------------------------------------------------------------------
    ridge = RidgeStacker()

    # Prepare Train Data for Ridge (Markdown Only)
    train_md_mask = (df_train["cell_type"] == "markdown").values
    X_train_ridge = tfidf_train[train_md_mask]
    y_train_ridge = df_train.loc[train_md_mask, "pct_rank"].values
    groups_train = df_train.loc[train_md_mask, "ancestor_id"].values

    # Generate OOF Predictions
    train_oof_preds = ridge.fit_predict_oof(
        X_train_ridge, y_train_ridge, groups_train, load_cached_data=True
    )

    # Predict on Validation
    val_md_mask = (df_val["cell_type"] == "markdown").values
    X_val_ridge = tfidf_val[val_md_mask]
    val_ridge_preds = ridge.predict(X_val_ridge)

    # --------------------------------------------------------------------------
    # 4. Feature Engineering (Anchors)
    # --------------------------------------------------------------------------
    anchor_eng = AnchorExtractor()

    # Extract Anchors
    # Note: We use a distinct cache name for the sampled train set to avoid conflict with full set
    df_train_anchors = anchor_eng.extract_features(
        df_train, tfidf_train, svd_train, "train_sampled", load_cached_data=True
    )
    df_val_anchors = anchor_eng.extract_features(
        df_val, tfidf_val, svd_val, "val", load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 5. Prepare Stage 2 Data
    # --------------------------------------------------------------------------
    # Merge Ridge Predictions
    # df_train_anchors is derived from iterating df_train's markdown cells in order.
    # train_oof_preds is derived from df_train[train_md_mask] in order.
    # They align perfectly.

    # Train
    df_stage2_train = df_train_anchors.copy()
    df_stage2_train["ridge_pred"] = train_oof_preds
    df_stage2_train = add_metadata_features(df_stage2_train, df_train)

    # Val
    df_stage2_val = df_val_anchors.copy()
    df_stage2_val["ridge_pred"] = val_ridge_preds
    df_stage2_val = add_metadata_features(df_stage2_val, df_val)

    # Define Feature Columns (exclude IDs)
    feature_cols = [c for c in df_stage2_train.columns if c not in ["cell_id", "id"]]

    # Targets
    y_train_lgbm = df_train.loc[train_md_mask, "pct_rank"].values
    y_val_lgbm = df_val.loc[val_md_mask, "pct_rank"].values

    # --------------------------------------------------------------------------
    # 6. Stage 2: LightGBM
    # --------------------------------------------------------------------------
    lgbm = LGBMRanker()
    lgbm.fit(
        df_stage2_train[feature_cols],
        y_train_lgbm,
        df_stage2_val[feature_cols],
        y_val_lgbm,
        load_cached_model=True,
    )

    # --------------------------------------------------------------------------
    # 7. Validation & Evaluation
    # --------------------------------------------------------------------------
    # Predict
    val_final_preds = lgbm.predict(df_stage2_val[feature_cols])

    # Prepare DataFrame for reconstruction
    df_val_preds = pd.DataFrame(
        {"cell_id": df_stage2_val["cell_id"], "pred_rank": val_final_preds}
    )

    # Reconstruct Orders
    df_val_orders = reconstruct_orders(df_val, df_val_preds)

    # Load Ground Truth
    df_val_truth = pd.read_csv(Config.VAL_METADATA_PATH)[["id", "cell_order"]]

    # Compute Metric
    score = kendall_tau_metric(df_val_truth, df_val_orders)
    print(f"Final Validation Metric: {score}")

    # --------------------------------------------------------------------------
    # 8. Failure Analysis
    # --------------------------------------------------------------------------
    # Calculate errors
    errors = np.abs(y_val_lgbm - val_final_preds)

    print("\n--- Failure Analysis (Correlation with Error) ---")
    # Check correlation with Ridge Preds
    corr_ridge, _ = pearsonr(errors, df_stage2_val["ridge_pred"])
    print(f"Ridge Prediction: {corr_ridge:.4f}")

    # Check correlation with Anchor Features (Top K Mean)
    if "lex_topk_mean" in df_stage2_val.columns:
        corr_lex, _ = pearsonr(errors, df_stage2_val["lex_topk_mean"])
        print(f"Lexical Top-K Mean: {corr_lex:.4f}")

    if "lat_topk_mean" in df_stage2_val.columns:
        corr_lat, _ = pearsonr(errors, df_stage2_val["lat_topk_mean"])
        print(f"Latent Top-K Mean: {corr_lat:.4f}")

    if "n_cells" in df_stage2_val.columns:
        corr_len, _ = pearsonr(errors, df_stage2_val["n_cells"])
        print(f"Notebook Length: {corr_len:.4f}")

    # --------------------------------------------------------------------------
    # 9. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7959051868218839

    if score > THRESHOLD:
        # Load Test Data
        df_test = dm.load_data("test", load_cached_data=True)

        # Transform
        tfidf_test, svd_test = vec.transform(
            df_test["source"].fillna("").astype(str).tolist()
        )

        # Ridge Predict
        test_md_mask = (df_test["cell_type"] == "markdown").values
        X_test_ridge = tfidf_test[test_md_mask]
        test_ridge_preds = ridge.predict(X_test_ridge)

        # Anchor Extract
        df_test_anchors = anchor_eng.extract_features(
            df_test, tfidf_test, svd_test, "test", load_cached_data=True
        )

        # Prepare Stage 2 Test Data
        df_stage2_test = df_test_anchors.copy()
        df_stage2_test["ridge_pred"] = test_ridge_preds
        df_stage2_test = add_metadata_features(df_stage2_test, df_test)

        # LGBM Predict
        test_final_preds = lgbm.predict(df_stage2_test[feature_cols])

        # Reconstruct Orders
        df_test_preds_map = pd.DataFrame(
            {"cell_id": df_stage2_test["cell_id"], "pred_rank": test_final_preds}
        )

        df_submission = reconstruct_orders(df_test, df_test_preds_map)

        # Save
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
