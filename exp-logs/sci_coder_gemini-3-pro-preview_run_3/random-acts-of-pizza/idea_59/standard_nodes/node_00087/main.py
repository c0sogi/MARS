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

    # --- PHASE 1: VALIDATION (TRAIN -> VAL) ---
    logger.info("=== PHASE 1: VALIDATION (TRAIN -> VAL) ===")

    # Split features into Train and Val subsets
    # Cite debug_lesson_11: Decouple Stacking OOF Generation from Hold-Out Evaluation
    X_train_dict = {k: v[train_indices] for k, v in X_union_dict.items()}
    y_train = y_union.iloc[train_indices]

    X_val_dict = {k: v[val_indices] for k, v in X_union_dict.items()}
    y_val = y_union.iloc[val_indices]

    # A. Run CV on Train ONLY
    # Cite debug_lesson_7: Avoid Static Cache Keys (use specific name for train OOF)
    oof_preds_train = pm.run_cv_and_oof(
        X_train_dict, y_train, load_cached_oof=False, cache_name="oof_train.npy"
    )

    # B. Train Meta-Learner on Train OOF
    # Cite debug_lesson_12: Isolate Validation Data During Meta-Learner Training
    pm.train_meta_learner(oof_preds_train, y_train)

    # C. Retrain Stable Models on Train ONLY
    pm.retrain_stable_full(X_train_dict, y_train)

    # D. Predict on Hold-out Validation Set
    val_final_preds = pm.predict(X_val_dict)

    # E. Compute Metric
    final_auc = roc_auc_score(y_val, val_final_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate Error Magnitude
    # For binary classification, error = |y_true - y_pred|
    errors = np.abs(y_val - val_final_preds)

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

    # 6. Submission
    THRESHOLD = 0.7222984867326668

    if final_auc > THRESHOLD:
        logger.info(
            f"Validation metric ({final_auc}) exceeds threshold ({THRESHOLD}). Proceeding to Submission Phase..."
        )

        # --- PHASE 2: SUBMISSION (UNION -> TEST) ---
        logger.info("=== PHASE 2: SUBMISSION (UNION -> TEST) ===")

        # A. Run CV on Full Union Dataset
        # This overwrites the models in 'models_dir' with ones trained on Union
        oof_preds_union = pm.run_cv_and_oof(
            X_union_dict, y_union, load_cached_oof=False, cache_name="oof_union.npy"
        )

        # B. Train Meta-Learner on Union OOF
        pm.train_meta_learner(oof_preds_union, y_union)

        # C. Retrain Stable Models on Full Union
        pm.retrain_stable_full(X_union_dict, y_union)

        # D. Predict on Test
        test_df = load_test_dataset(load_cached_data=True)
        test_ids = test_df[ID_COL].values
        X_test_dict = fe_pipeline.transform(test_df, load_cached_data=True)

        pm.predict_and_submit(X_test_dict, test_ids)

    else:
        logger.warning(
            f"Validation metric ({final_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
