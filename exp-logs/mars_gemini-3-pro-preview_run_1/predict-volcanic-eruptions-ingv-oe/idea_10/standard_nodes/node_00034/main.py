import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Import library modules
from library.config import Config
from library.train_lgbm import run_lgbm_cv
from library.train_cnn import run_cnn_cv
from library.meta_learner import train_ridge_stacker
from library.dataset import VolcanoTabularBuilder
from library.utils import seed_everything, get_logger


def main():
    # 1. Setup and Configuration Overrides for Fast Baseline
    logger = get_logger("runfile")
    seed_everything(Config.SEED)

    logger.info("Starting Runfile Execution...")

    # Override Config for speed (Fast Baseline)
    # LightGBM: Reduce estimators
    Config.LGB_PARAMS["n_estimators"] = 2000
    Config.LGB_PARAMS["early_stopping_rounds"] = 50

    # CNN: Reduce epochs, increase batch size for A100
    Config.CNN_PARAMS["epochs"] = 15
    Config.CNN_PARAMS["batch_size"] = 64

    logger.info(
        f"Config overrides applied: LGB n_estimators={Config.LGB_PARAMS['n_estimators']}, "
        f"CNN epochs={Config.CNN_PARAMS['epochs']}"
    )

    # 2. Execute Training Pipelines

    # Branch A: LightGBM
    logger.info(">>> Running LightGBM Pipeline...")
    lgb_oof, lgb_test = run_lgbm_cv(debug=False)

    # Branch B: CNN
    logger.info(">>> Running CNN Pipeline...")
    cnn_oof, cnn_test = run_cnn_cv(debug=False)

    # Meta-Learner: Stacking
    logger.info(">>> Running Meta-Learner Stacking...")
    submission_df, meta_model = train_ridge_stacker(
        lgb_oof_df=lgb_oof,
        lgb_test_df=lgb_test,
        cnn_oof_df=cnn_oof,
        cnn_test_df=cnn_test,
    )

    # 3. Validation Assessment
    logger.info(">>> Performing Validation Assessment...")

    # Re-construct OOF dataset for Meta-Learner validation
    # Merge LGB and CNN OOFs
    lgb_oof_renamed = lgb_oof.rename(columns={"pred": "pred_lgb"})
    cnn_oof_renamed = cnn_oof.rename(columns={"pred": "pred_cnn"})

    meta_val_df = pd.merge(
        lgb_oof_renamed[["segment_id", "time_to_eruption", "pred_lgb"]],
        cnn_oof_renamed[["segment_id", "pred_cnn"]],
        on="segment_id",
        how="inner",
    )

    X_meta_val = meta_val_df[["pred_lgb", "pred_cnn"]].values
    y_meta_val = meta_val_df["time_to_eruption"].values

    # Predict using the trained meta_model
    y_pred_meta = meta_model.predict(X_meta_val)

    # Calculate Final Metric
    final_mae = mean_absolute_error(y_meta_val, y_pred_meta)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_mae}")

    # 4. Failure Analysis
    logger.info(">>> Performing Failure Analysis...")

    # Calculate Absolute Errors
    meta_val_df["abs_error"] = np.abs(meta_val_df["time_to_eruption"] - y_pred_meta)

    # Load Tabular Features to correlate with error
    builder = VolcanoTabularBuilder()
    # Load both train and val to match the CV OOF set
    df_train, _, _ = builder.get_data("train", load_cache=True)
    df_val, _, _ = builder.get_data("val", load_cache=True)
    df_features = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Merge features with errors
    analysis_df = pd.merge(
        meta_val_df[["segment_id", "abs_error"]],
        df_features,
        on="segment_id",
        how="inner",
    )

    # Calculate Correlations
    # Drop non-numeric or ID columns
    drop_cols = ["segment_id", "time_to_eruption", "abs_error"]
    feature_cols = [c for c in analysis_df.columns if c not in drop_cols]

    correlations = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(analysis_df[col]):
            corr = analysis_df[col].corr(analysis_df["abs_error"])
            correlations[col] = corr

    # Sort and Print Top Correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n--- Failure Analysis: Top 10 Features Correlated with Error ---")
    for feat, corr in sorted_corr[:10]:
        print(f"{feat}: {corr:.4f}")

    # 5. Submission Logic
    # Threshold: 2250276.65
    THRESHOLD = 2250276.65
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if final_mae < THRESHOLD:
        logger.info(
            f"Validation Metric ({final_mae}) meets threshold ({THRESHOLD}). Keeping submission."
        )
        # Ensure file exists (train_ridge_stacker saves it, but we double check)
        if not os.path.exists(submission_path):
            submission_df.to_csv(submission_path, index=False)
    else:
        logger.warning(
            f"Validation Metric ({final_mae}) does NOT meet threshold ({THRESHOLD}). Removing submission."
        )
        if os.path.exists(submission_path):
            os.remove(submission_path)
            logger.info("Submission file removed.")


if __name__ == "__main__":
    main()
