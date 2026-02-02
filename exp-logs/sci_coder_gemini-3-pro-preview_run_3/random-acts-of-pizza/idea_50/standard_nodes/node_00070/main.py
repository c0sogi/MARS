import os
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import roc_auc_score

from library.config import (
    SEED,
    WORKING_DIR,
    SUBMISSION_FILE,
    TARGET_COL,
    ID_COL,
    N_FOLDS,
)
from library.utils import set_seed, Timer
from library.data_loader import load_dataset
from library.features import FeatureEngineer
from library.pipeline import HybridStackingRunner


def main():
    # 1. Setup
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    print("Initializing Hex-View Stacking Ensemble Pipeline...")

    # 2. Load Data
    # val_df here is the official hold-out validation set (from metadata/val.parquet)
    # We must ensure this is NOT passed to the training pipeline to prevent leakage.
    train_df, val_df, test_df = load_dataset(load_cached_data=True)

    # 3. Feature Engineering
    # We fit vectorizers on train_df and transform all sets.
    fe = FeatureEngineer()
    train_feats, val_feats, test_feats = fe.generate_features(train_df, val_df, test_df)

    train_y = train_df[TARGET_COL].values
    val_y = val_df[TARGET_COL].values
    test_ids = test_df[ID_COL].values

    # 4. Prepare Data for Stacking Runner
    # The HybridStackingRunner.run_stacking method merges its 'train' and 'val' arguments
    # to perform internal Cross-Validation.
    # To utilize the full 'train_df' for training while keeping 'val_df' strictly for
    # hold-out evaluation, we split 'train_feats' into two chunks.
    # The runner will merge them back together, effectively training on the full train set.
    split_idx = int(len(train_y) * 0.9)

    stack_train_feats = {}
    stack_val_feats = {}

    # Slice features (handling both numpy arrays and sparse matrices)
    for k, v in train_feats.items():
        stack_train_feats[k] = v[:split_idx]
        stack_val_feats[k] = v[split_idx:]

    stack_train_y = train_y[:split_idx]
    stack_val_y = train_y[split_idx:]

    # 5. Run Training Pipeline
    # This trains models, saves them to disk, and generates the submission for test_df
    runner = HybridStackingRunner()
    runner.run_stacking(
        stack_train_feats,
        stack_val_feats,
        test_feats,
        stack_train_y,
        stack_val_y,
        test_ids,
    )

    # 6. Validation Inference (Hold-out)
    print("\n" + "=" * 10 + " Hold-out Validation " + "=" * 10)

    # Load Meta-Learner
    model_dir = os.path.join(WORKING_DIR, "models")
    meta_learner = joblib.load(os.path.join(model_dir, "meta_learner.joblib"))

    # Define Model Configuration (Must match order in models.py/pipeline.py)
    # Volatile models use CV-Bagging (Average of 5 folds)
    # Stable models use the single model retrained on full data
    volatile_models = ["semantic_booster", "temporal_booster"]
    model_order = [
        "lexical_bagger",
        "community_bagger",
        "semantic_booster",
        "semantic_bagger",
        "metadata_anchor",
        "temporal_booster",
    ]

    # Generate Level 1 Predictions on Hold-out Set
    l1_preds = np.zeros((len(val_y), len(model_order)))

    with Timer("Validation Inference"):
        for i, name in enumerate(model_order):
            if name in volatile_models:
                # Average predictions from all 5 fold models
                fold_preds = []
                for fold in range(N_FOLDS):
                    model_path = os.path.join(model_dir, f"{name}_fold_{fold}.joblib")
                    model = joblib.load(model_path)
                    fold_preds.append(model.predict_proba(val_feats)[:, 1])
                l1_preds[:, i] = np.mean(fold_preds, axis=0)
            else:
                # Use single fully-retrained model
                model_path = os.path.join(model_dir, f"{name}.joblib")
                model = joblib.load(model_path)
                l1_preds[:, i] = model.predict_proba(val_feats)[:, 1]

    # Generate Final Predictions via Meta-Learner
    val_final_preds = meta_learner.predict_proba(l1_preds)[:, 1]

    # Compute and Print Metric
    val_auc = roc_auc_score(val_y, val_final_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\nFailure Analysis on Validation Set:")
    # Calculate absolute error
    errors = np.abs(val_y - val_final_preds)

    # Analyze correlation with numerical metadata features
    analysis_df = val_df.select_dtypes(include=[np.number]).copy()
    # Drop target and irrelevant IDs if present
    cols_to_drop = [c for c in [TARGET_COL, ID_COL] if c in analysis_df.columns]
    analysis_df = analysis_df.drop(columns=cols_to_drop, errors="ignore")

    analysis_df["error_magnitude"] = errors

    # Compute correlations
    corrs = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 5 Features Correlated with Prediction Error:")
    print(corrs.head(5))

    # 8. Conditional Submission Logic
    threshold = 0.7138293787137718
    if val_auc <= threshold:
        print(
            f"\n[Check] Validation metric {val_auc} <= {threshold}. Deleting submission file."
        )
        if os.path.exists(SUBMISSION_FILE):
            os.remove(SUBMISSION_FILE)
    else:
        print(
            f"\n[Check] Validation metric {val_auc} > {threshold}. Submission file retained."
        )


if __name__ == "__main__":
    main()
