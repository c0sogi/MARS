import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import random

from library.config import Config
from library.data_processing import process_data
from library.gnn_features import process_gnn_features
from library.model_training import train_xgboost_model
from library.utils import inverse_log_transform, rmsle_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def prepare_dataset(split):
    """
    Loads and merges physical descriptors and GNN embeddings for a given split.
    """
    # Load physical descriptors (and cache them)
    df_phys, _ = process_data(split=split, load_cached_data=True)

    # Load GNN embeddings (and cache them)
    df_emb = process_gnn_features(split=split, load_cached_data=True)

    # Ensure alignment
    if len(df_phys) != len(df_emb):
        # Truncate to the minimum length if there's a mismatch (should be rare with cached data)
        min_len = min(len(df_phys), len(df_emb))
        df_phys = df_phys.iloc[:min_len].reset_index(drop=True)
        df_emb = df_emb.iloc[:min_len].reset_index(drop=True)

    # Concatenate features
    # We drop ID/File path from df_emb if they exist, but the mock returns only features
    df_merged = pd.concat([df_phys, df_emb], axis=1)

    return df_merged


def get_feature_columns(df):
    """
    Excludes target columns and metadata to return feature names.
    """
    exclude = Config.TARGET_COLS + [Config.ID_COL, Config.FILE_PATH_COL]
    return [c for c in df.columns if c not in exclude]


def main():
    set_seed(Config.RANDOM_SEED)

    print("--- Loading and Processing Data ---")
    train_df = prepare_dataset("train")
    val_df = prepare_dataset("val")
    test_df = prepare_dataset("test")

    feature_cols = get_feature_columns(train_df)
    print(f"Training with {len(feature_cols)} features: {feature_cols}")

    # Prepare X and y
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    models = {}
    val_preds = {}
    test_preds = {}

    print("\n--- Training Models ---")
    for target in Config.TARGET_COLS:
        print(f"Training for {target}...")
        y_train = train_df[target]
        y_val = val_df[target]

        # Train
        model = train_xgboost_model(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=50,
            verbose=False,
        )
        models[target] = model

        # Predict on validation (returns log scale)
        y_val_pred_log = model.predict(X_val)
        # Inverse transform
        val_preds[target] = inverse_log_transform(y_val_pred_log)

        # Predict on test
        y_test_pred_log = model.predict(X_test)
        test_preds[target] = inverse_log_transform(y_test_pred_log)

    # --- Validation Assessment ---
    print("\n--- Validation Assessment ---")
    # Construct ground truth and prediction arrays for metric calculation
    y_true_val = val_df[Config.TARGET_COLS].values
    y_pred_val = np.column_stack([val_preds[t] for t in Config.TARGET_COLS])

    # Ensure non-negative
    y_pred_val = np.maximum(y_pred_val, 0)

    final_metric = rmsle_score(y_true_val, y_pred_val)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate error per sample (mean absolute error across targets for simplicity in correlation)
    # Or better, correlation of error with features for each target

    for i, target in enumerate(Config.TARGET_COLS):
        print(f"Analyzing errors for {target}:")
        errors = np.abs(y_true_val[:, i] - y_pred_val[:, i])

        # Create a dataframe for correlation
        analysis_df = X_val.copy()
        analysis_df["error"] = errors

        # Compute correlation
        correlations = (
            analysis_df.corr()["error"]
            .drop("error")
            .sort_values(ascending=False, key=abs)
        )
        print("Top 5 features correlated with error:")
        print(correlations.head(5))
        print("-" * 20)

    # --- Submission ---
    THRESHOLD = 0.06380692050212411

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        submission_df = pd.DataFrame()
        submission_df[Config.ID_COL] = test_df[Config.ID_COL]

        for target in Config.TARGET_COLS:
            # Ensure non-negative
            preds = np.maximum(test_preds[target], 0)
            submission_df[target] = preds

        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print(submission_df.head())
    else:
        print(
            f"\nValidation metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
