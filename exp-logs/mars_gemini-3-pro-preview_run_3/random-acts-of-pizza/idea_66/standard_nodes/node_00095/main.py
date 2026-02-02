import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Configuration Monkey-Patching for Speed
# -----------------------------------------------------------------------------
from library.config import Config

# Reduce estimators for fast baseline execution to ensure < 42 min runtime
Config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 50
Config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 50
Config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 100
Config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 100
Config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 50
Config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 100
Config.LEXICAL_BAGGER_PARAMS["n_jobs"] = 12
Config.COMMUNITY_BAGGER_PARAMS["n_jobs"] = 12
Config.SEMANTIC_BOOSTER_PARAMS["n_jobs"] = 12
Config.SEMANTIC_GRADIENT_PARAMS["n_jobs"] = 12

# -----------------------------------------------------------------------------
# 2. Import Library Modules
# -----------------------------------------------------------------------------
from library.pipeline import (
    run_training_pipeline,
    run_inference_pipeline,
    _get_feature_set_for_model,
    _is_volatile_model,
)
from library.data_loader import load_dataset, _process_dataframe
from library.feature_engineering import FeatureFactory
from library.model_factory import ModelFactory
from library.trainer import Trainer

# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------


def main():
    # Set seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    print("=== Starting Runfile ===")

    # -------------------------------------------------------------------------
    # Step 1: Training
    # -------------------------------------------------------------------------
    print("\n[Step 1] Running Training Pipeline...")
    # We use load_cached_data=False to ensure we start fresh with our patched config
    run_training_pipeline(load_cached_data=False)

    # -------------------------------------------------------------------------
    # Step 2: Validation on Hold-Out Set
    # -------------------------------------------------------------------------
    print("\n[Step 2] Performing Validation on Hold-Out Set...")

    # A. Load Data
    # We need the training union data to fit the feature extractors (TF-IDF, etc.)
    # and the validation data to transform and predict.
    # load_dataset returns (train_union, test). We discard test for now.
    train_union, _ = load_dataset(load_cached_data=True)

    # Load raw validation data manually
    if not os.path.exists(Config.VAL_DATA_PATH):
        raise FileNotFoundError(f"Val data not found at {Config.VAL_DATA_PATH}")
    raw_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Process validation data (clean text, etc.)
    # We process with is_test=False to preserve the target column for evaluation
    val_df = _process_dataframe(raw_val, is_test=False)
    y_val = val_df[Config.TARGET_COL].values

    # B. Generate Features for Validation
    # We must re-fit on train_union and transform val_df.
    # IMPORTANT: This will overwrite the 'X_test' cache files in ./working/idea_XX/
    # We pass load_cached_data=False to force re-computation for the new 'test' (which is val)
    print("Generating features for validation set...")
    ff = FeatureFactory()

    # We ignore the returned X_train as we only need X_val (returned as second arg)
    _, X_val_lex = ff.make_lexical(train_union, val_df, load_cached_data=False)
    _, X_val_com = ff.make_behavioral(train_union, val_df, load_cached_data=False)
    _, X_val_sem = ff.make_semantic(train_union, val_df, load_cached_data=False)
    _, X_val_meta = ff.make_metadata(train_union, val_df, load_cached_data=False)

    features_val = {
        "lexical": X_val_lex,
        "community": X_val_com,
        "semantic": X_val_sem,
        "meta": X_val_meta,
    }

    # C. Predict
    print("Predicting on validation set...")
    trainer = Trainer()
    base_models = ModelFactory.get_base_models()
    level1_preds = pd.DataFrame(index=range(len(val_df)))

    for model_name in base_models.keys():
        X_main, X_meta = _get_feature_set_for_model(model_name, features_val)
        X_val_fold = trainer._concat_features(X_main, X_meta)

        if _is_volatile_model(model_name):
            # Hybrid Inference: Average of 5 fold models
            fold_preds = []
            for fold in range(Config.N_FOLDS):
                path = trainer._get_model_path(model_name, fold)
                model = joblib.load(path)
                pred = model.predict_proba(X_val_fold)[:, 1]
                fold_preds.append(pred)
            level1_preds[model_name] = np.mean(fold_preds, axis=0)
        else:
            # Stable: Use full model
            path = trainer._get_model_path(model_name)
            model = joblib.load(path)
            level1_preds[model_name] = model.predict_proba(X_val_fold)[:, 1]

    # Level 2 Prediction
    meta_path = trainer._get_model_path("meta_learner")
    meta_model = joblib.load(meta_path)
    val_final_preds = meta_model.predict_proba(level1_preds.values)[:, 1]

    # D. Metric
    final_metric = roc_auc_score(y_val, val_final_preds)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # Step 3: Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 3] Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(y_val - val_final_preds)

    # Correlate with metadata features
    analysis_cols = [c for c in Config.METADATA_COLS if c in val_df.columns]
    print("Correlation between Error and Features:")
    for col in analysis_cols:
        if val_df[col].nunique() > 1:
            corr, _ = pearsonr(errors, val_df[col])
            print(f"  {col}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # Step 4: Submission
    # -------------------------------------------------------------------------
    threshold = 0.7222984867326668
    if final_metric > threshold:
        print(
            f"\n[Step 4] Metric ({final_metric}) > Threshold ({threshold}). Generating Submission..."
        )
        # CRITICAL: We must run inference pipeline with load_cached_data=False
        # because we overwrote the 'X_test' cache with validation data in Step 2.
        # This forces the FeatureFactory to re-generate features for the ACTUAL test set.
        run_inference_pipeline(load_cached_data=False)
    else:
        print(
            f"\n[Step 4] Metric ({final_metric}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
