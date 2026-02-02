import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_and_clean_data
from library.feature_engineering import FeatureEngineer
from library.model_architecture import PentViewEnsemble


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    # Enable GPU for XGBoost if available
    if torch.cuda.is_available():
        Config.XGB_PARAMS["device"] = "cuda"
        logger.info("GPU detected. XGBoost will use CUDA.")

    # 2. Load Data
    logger.info("Loading data...")
    train_df, val_df, test_df = load_and_clean_data(load_cached_data=True)

    # Tag validation data to extract it later for specific reporting
    train_df["is_validation"] = False
    val_df["is_validation"] = True

    # Combine for CV-Bagging Strategy
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    y = full_df[Config.TARGET_COL].values

    # 3. Cross-Validation Training Loop
    logger.info("Starting CV-Bagging Training...")

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    oof_preds = np.zeros(len(full_df))
    models_dir = os.path.join(Config.CACHE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y)):
        logger.info(f"--- Fold {fold + 1} / {Config.N_FOLDS} ---")

        # Split Data
        fold_train_df = full_df.iloc[train_idx].copy()
        fold_val_df = full_df.iloc[val_idx].copy()
        y_fold_train = y[train_idx]
        y_fold_val = y[val_idx]

        # Feature Engineering (Fit on Fold Train)
        fe = FeatureEngineer()
        fe.fit(fold_train_df)

        # Transform Views
        train_views = fe.transform(
            fold_train_df, split_name=f"fold_{fold}_train", load_cache=True
        )
        val_views = fe.transform(
            fold_val_df, split_name=f"fold_{fold}_val", load_cache=True
        )

        # Model Training
        model = PentViewEnsemble()
        model.fit(train_views, y_fold_train, val_views, y_fold_val)

        # Inference on Fold Validation
        val_preds = model.predict_proba(val_views)
        oof_preds[val_idx] = val_preds

        # Save Artifacts
        joblib.dump(fe, os.path.join(models_dir, f"fe_fold_{fold}.joblib"))
        joblib.dump(model, os.path.join(models_dir, f"model_fold_{fold}.joblib"))

    # 4. Validation Metric Calculation
    # We extract the predictions corresponding to the original 'val.parquet' rows
    val_mask = full_df["is_validation"].values
    val_preds_holdout = oof_preds[val_mask]
    val_targets_holdout = y[val_mask]

    final_metric = roc_auc_score(val_targets_holdout, val_preds_holdout)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate Error
    errors = np.abs(val_targets_holdout - val_preds_holdout)

    # Prepare DataFrame for correlation
    analysis_df = full_df[val_mask].copy()
    analysis_df["prediction_error"] = errors

    # Correlate with Numerical Features
    logger.info("Correlation between Prediction Error and Numerical Features:")
    correlations = {}
    for col in Config.NUMERICAL_FEATURES:
        if col in analysis_df.columns:
            # Handle NaNs just in case
            series = analysis_df[col].fillna(analysis_df[col].median())
            if series.nunique() > 1:
                corr, _ = pearsonr(series, analysis_df["prediction_error"])
                correlations[col] = corr
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: Constant (Skipped)")

    # 6. Submission Generation
    THRESHOLD = 0.7085870249842536

    if final_metric > THRESHOLD:
        logger.info(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        bagged_test_preds = np.zeros(len(test_df))

        for fold in range(Config.N_FOLDS):
            # Load Artifacts
            fe = joblib.load(os.path.join(models_dir, f"fe_fold_{fold}.joblib"))
            model = joblib.load(os.path.join(models_dir, f"model_fold_{fold}.joblib"))

            # Transform Test Data
            test_views = fe.transform(
                test_df, split_name=f"fold_{fold}_test", load_cache=True
            )

            # Predict
            preds = model.predict_proba(test_views)
            bagged_test_preds += preds

        # Average Predictions
        avg_preds = bagged_test_preds / Config.N_FOLDS

        # Save
        submission = pd.DataFrame(
            {"request_id": test_df[Config.ID_COL], Config.TARGET_COL: avg_preds}
        )

        submission.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_OUTPUT_PATH}")

    else:
        logger.warning(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
