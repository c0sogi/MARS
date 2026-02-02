import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import SEED, ID_COL, TARGET_COL, TRAIN_PATH, VAL_PATH, WORKING_DIR
from library.utils import set_seed, get_logger, print_metrics
from library.data_factory import load_union_dataset, load_test_dataset
from library.feature_engineering import FeaturePipeline
from library.pipeline_manager import PipelineManager

# Initialize Logger
logger = get_logger("runfile")


def main():
    # 1. Setup
    set_seed(SEED)
    logger.info("Starting execution of Hept-View Stacking Ensemble...")

    # 2. Load Data
    # We load the Union Dataset (Train + Val) for the pipeline
    union_df = load_union_dataset(load_cached_data=True)

    # We need to identify which rows belong to the original validation set for reporting
    # The data_factory concatenates [train, val].
    # Let's load the raw metadata files briefly to get the lengths/indices.
    train_meta = pd.read_parquet(TRAIN_PATH)
    val_meta = pd.read_parquet(VAL_PATH)

    n_train = len(train_meta)
    n_val = len(val_meta)

    # Validation indices in the union dataframe are the last n_val rows
    train_indices = np.arange(0, n_train)
    val_indices = np.arange(n_train, n_train + n_val)

    # Verify alignment
    assert len(union_df) == n_train + n_val
    assert union_df.iloc[val_indices[0]][ID_COL] == val_meta.iloc[0][ID_COL]
    logger.info(
        f"Data alignment verified. Validation set starts at index {val_indices[0]}."
    )

    # 3. Feature Engineering
    # Initialize pipeline
    fe_pipeline = FeaturePipeline()

    # Generate/Load features for Union Dataset
    # This handles caching automatically
    X_union_dict = fe_pipeline.fit_transform(union_df, load_cached_data=True)
    y_union = union_df[TARGET_COL]

    # 4. Pipeline Execution
    pm = PipelineManager()

    # A. Run CV and Generate OOF Predictions on Union Dataset
    # This trains models on 5 folds and produces predictions for every sample in union_df
    oof_preds_union = pm.run_cv_and_oof(X_union_dict, y_union, load_cached_oof=True)

    # B. Train Meta-Learner on Union OOF
    # Cite debug_lesson_12: Isolate Validation Data During Meta-Learner Training
    meta_learner = pm.train_meta_learner(
        oof_preds_union[train_indices], y_union.iloc[train_indices]
    )

    # C. Retrain Stable Models on Full Union Dataset
    pm.retrain_stable_full(X_union_dict, y_union)

    # 5. Validation Assessment (Strictly on Hold-out Validation Set)
    logger.info("Performing Validation Assessment...")

    # Extract OOF predictions for the validation subset
    # oof_preds_union is (n_samples, n_models)
    # We need the Meta-Learner's prediction on these OOF inputs

    # Get Level 1 OOF predictions for validation rows
    val_L1_oof = oof_preds_union[val_indices]
    y_val_true = y_union.iloc[val_indices].values

    # Predict using Meta-Learner
    val_final_preds = meta_learner.predict_proba(val_L1_oof)

    # Compute Metric
    final_auc = roc_auc_score(y_val_true, val_final_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate Error Magnitude
    # For binary classification, error = |y_true - y_pred|
    errors = np.abs(y_val_true - val_final_preds)

    # Correlate with Metadata Features
    # We use the scaled metadata features from X_union_dict corresponding to validation
    X_val_meta = X_union_dict["metadata"][val_indices]

    # We need to map back to feature names.
    # Since FeaturePipeline uses StandardScaler on ALLOW_LIST, order is preserved.
    # We need the list of columns actually used.
    from library.config import ALLOW_LIST

    used_cols = [c for c in ALLOW_LIST if c in union_df.columns]

    correlations = []
    for i, col_name in enumerate(used_cols):
        # Calculate Pearson correlation between feature value and error
        feature_vals = X_val_meta[:, i]
        corr, _ = pearsonr(feature_vals, errors)
        correlations.append((col_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n--- Failure Analysis: Feature-Error Correlations ---")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # Also check correlation with text length (re-calculated from raw df)
    val_subset_df = union_df.iloc[val_indices]
    # Use 'request_text_edit_aware' as defined in config TEXT_COLS
    text_col = "request_text_edit_aware"
    if text_col in val_subset_df.columns:
        lengths = val_subset_df[text_col].fillna("").astype(str).apply(len)
        corr_len, _ = pearsonr(lengths, errors)
        print(f"Text Length (Chars): {corr_len:.4f}")

    # 7. Submission
    THRESHOLD = 0.7222984867326668

    if final_auc > THRESHOLD:
        logger.info(
            f"Validation metric ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_df = load_test_dataset(load_cached_data=True)
        test_ids = test_df[ID_COL].values

        # Generate Test Features
        # Note: transform uses the transformers fitted on Union Dataset
        X_test_dict = fe_pipeline.transform(test_df, load_cached_data=True)

        # Predict and Submit
        pm.predict_and_submit(X_test_dict, test_ids)

    else:
        logger.warning(
            f"Validation metric ({final_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
