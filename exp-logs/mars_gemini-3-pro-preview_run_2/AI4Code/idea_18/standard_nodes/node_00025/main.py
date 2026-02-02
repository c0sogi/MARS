import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Import provided library components
from library.config import Config
from library.utils import kendall_tau, format_submission
from library.data_processing import NotebookProcessor
from library.feature_extraction import FeatureEngine
from library.modeling import Stage1Ridge, Stage2LGBM


def main():
    # ------------------------------------------------------------------
    # 1. Setup and Configuration
    # ------------------------------------------------------------------
    Config.set_seeds()
    Config.setup()

    # Dynamic GPU Configuration
    if torch.cuda.is_available():
        print("GPU detected. Configuring LightGBM to use GPU.")
        Config.LGBM_PARAMS["device"] = "gpu"
        Config.LGBM_PARAMS["gpu_platform_id"] = 0
        Config.LGBM_PARAMS["gpu_device_id"] = 0
    else:
        print("No GPU detected. Using CPU.")

    # Initialize Components
    processor = NotebookProcessor()
    feature_engine = FeatureEngine()
    stage1_model = Stage1Ridge()
    stage2_model = Stage2LGBM()

    # ------------------------------------------------------------------
    # 2. Data Loading and Subsampling (Fast Baseline)
    # ------------------------------------------------------------------
    print("Loading datasets...")
    # Load full training metadata
    df_train_full = processor.load_dataset(split="train", load_cached_data=True)

    # Load full validation data (Must use full set for valid metric)
    df_val = processor.load_dataset(split="val", load_cached_data=True)

    # Subsample Training Data for Speed
    # Target: ~30,000 notebooks to ensure completion within 2 hours
    TARGET_TRAIN_SIZE = 30000
    unique_ancestors = df_train_full["ancestor_id"].unique()

    if len(unique_ancestors) > TARGET_TRAIN_SIZE:
        print(f"Subsampling training data to {TARGET_TRAIN_SIZE} groups...")
        selected_ancestors = np.random.choice(
            unique_ancestors, size=TARGET_TRAIN_SIZE, replace=False
        )
        df_train = df_train_full[
            df_train_full["ancestor_id"].isin(selected_ancestors)
        ].reset_index(drop=True)
    else:
        df_train = df_train_full

    print(
        f"Training on {len(df_train)} cells from {df_train['ancestor_id'].nunique()} notebooks."
    )
    print(
        f"Validating on {len(df_val)} cells from {df_val['ancestor_id'].nunique()} notebooks."
    )

    # Filter for markdown cells (targets)
    train_md = df_train[df_train["cell_type"] == "markdown"].reset_index(drop=True)
    val_md = df_val[df_val["cell_type"] == "markdown"].reset_index(drop=True)

    y_train = train_md["pct_rank"].values
    y_val = val_md["pct_rank"].values

    # ------------------------------------------------------------------
    # 3. Feature Engineering
    # ------------------------------------------------------------------
    # Fit Global Vectorizers on the subsampled training data
    # We disable cache loading for vectorizers to ensure they match the subsample
    feature_engine.fit_global_vectorizers(df_train, load_cached_models=False)

    # ------------------------------------------------------------------
    # 4. Stage 1: Ridge Regression Stacking
    # ------------------------------------------------------------------
    print("--- Stage 1: Ridge Regression ---")

    # Generate OOF Predictions
    X_train_sparse = feature_engine.get_stage1_features(train_md)
    groups = train_md["ancestor_id"]

    # 5-Fold Group CV
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(train_md))

    print("Generating OOF predictions...")
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X_train_sparse, y_train, groups)
    ):
        X_fold_train = X_train_sparse[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_sparse[val_idx]

        model = Ridge(alpha=Config.RIDGE_ALPHA, random_state=Config.SEED)
        model.fit(X_fold_train, y_fold_train)
        oof_preds[val_idx] = model.predict(X_fold_val)

    # Fit Final Stage 1 Model
    stage1_model.fit(X_train_sparse, y_train)

    # Predict on Validation
    X_val_sparse = feature_engine.get_stage1_features(val_md)
    val_stage1_preds = stage1_model.predict(X_val_sparse)

    # ------------------------------------------------------------------
    # 5. Stage 2: LightGBM Refinement
    # ------------------------------------------------------------------
    print("--- Stage 2: LightGBM ---")

    # Generate Features
    # Note: We do NOT load cached data for train to ensure it matches our subsample
    print("Generating Stage 2 features for training (Subsampled)...")
    X_train_lgbm = feature_engine.get_stage2_features(
        df_train, split="train", load_cached_data=False
    )

    print("Generating Stage 2 features for validation...")
    X_val_lgbm = feature_engine.get_stage2_features(
        df_val, split="val", load_cached_data=True
    )

    # Add Stage 1 Predictions
    X_train_lgbm["ridge_pred"] = oof_preds
    X_val_lgbm["ridge_pred"] = val_stage1_preds

    # Prepare for LGBM
    drop_cols = ["id", "cell_id"]
    features_train = X_train_lgbm.drop(columns=drop_cols, errors="ignore")
    features_val = X_val_lgbm.drop(columns=drop_cols, errors="ignore")

    # Fit Model
    stage2_model.fit(features_train, y_train, features_val, y_val)

    # ------------------------------------------------------------------
    # 6. Validation and Metric Calculation
    # ------------------------------------------------------------------
    print("--- Evaluation ---")
    final_val_preds = stage2_model.predict(features_val)

    # Reconstruct Orders
    val_ids = val_md["id"].unique()
    pred_orders = []

    # Create lookup df
    val_md_preds = val_md.copy()
    val_md_preds["pred_rank"] = final_val_preds

    df_val_code = df_val[df_val["cell_type"] == "code"]

    print("Reconstructing validation orders...")
    # Optimization: Process groups using pandas groupby instead of loop
    # but for safety and clarity using the logic from workflow.py
    for nb_id in val_ids:
        nb_code = df_val_code[df_val_code["id"] == nb_id].copy()
        nb_md = val_md_preds[val_md_preds["id"] == nb_id].copy()

        n_code = len(nb_code)
        if n_code > 0:
            nb_code["score"] = np.linspace(0, 1, n_code)
        else:
            nb_code["score"] = []

        nb_md["score"] = nb_md["pred_rank"]

        nb_full = pd.concat([nb_code, nb_md]).sort_values("score")
        pred_orders.append(" ".join(nb_full["cell_id"].tolist()))

    pred_df = pd.DataFrame({"id": val_ids, "cell_order": pred_orders})
    gt_df = pd.read_csv(Config.VAL_METADATA_PATH)[["id", "cell_order"]]

    metric_score = kendall_tau(gt_df, pred_df)
    print(f"Final Validation Metric: {metric_score}")

    # ------------------------------------------------------------------
    # 7. Failure Analysis
    # ------------------------------------------------------------------
    print("--- Failure Analysis ---")
    # Calculate absolute error
    analysis_df = features_val.copy()
    analysis_df["target"] = y_val
    analysis_df["prediction"] = final_val_preds
    analysis_df["error"] = (analysis_df["target"] - analysis_df["prediction"]).abs()

    # Compute correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)

    print("Top 5 features correlated with Error:")
    print(correlations.head(6).iloc[1:])  # Skip 'error' itself

    # ------------------------------------------------------------------
    # 8. Submission
    # ------------------------------------------------------------------
    THRESHOLD = 0.7959051868218839

    if metric_score > THRESHOLD:
        print(
            f"Metric ({metric_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        df_test = processor.load_dataset(split="test", load_cached_data=True)
        test_md = df_test[df_test["cell_type"] == "markdown"].reset_index(drop=True)

        # Stage 1 Preds
        X_test_sparse = feature_engine.get_stage1_features(test_md)
        stage1_preds = stage1_model.predict(X_test_sparse)

        # Stage 2 Features
        X_test_lgbm = feature_engine.get_stage2_features(
            df_test, split="test", load_cached_data=True
        )
        X_test_lgbm["ridge_pred"] = stage1_preds

        features_test = X_test_lgbm.drop(columns=drop_cols, errors="ignore")

        # Final Preds
        final_test_preds = stage2_model.predict(features_test)

        # Sorting
        test_md_preds = test_md.copy()
        test_md_preds["pred_rank"] = final_test_preds
        df_test_code = df_test[df_test["cell_type"] == "code"]

        submission_ids = []
        submission_orders = []

        test_ids = df_test["id"].unique()

        for nb_id in test_ids:
            nb_code = df_test_code[df_test_code["id"] == nb_id].copy()
            nb_md = test_md_preds[test_md_preds["id"] == nb_id].copy()

            n_code = len(nb_code)
            if n_code > 0:
                nb_code["score"] = np.linspace(0, 1, n_code)
            else:
                nb_code["score"] = []

            nb_md["score"] = nb_md["pred_rank"]

            nb_full = pd.concat([nb_code, nb_md]).sort_values("score")
            submission_ids.append(nb_id)
            submission_orders.append(" ".join(nb_full["cell_id"].tolist()))

        sub_df = pd.DataFrame({"id": submission_ids, "cell_order": submission_orders})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Metric ({metric_score}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
