import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("meta_learner")


def train_ridge_stacker(
    lgb_oof_df=None, lgb_test_df=None, cnn_oof_df=None, cnn_test_df=None
):
    """
    Trains a Ridge Regression Meta-Learner on the OOF predictions from Branch A (LightGBM)
    and Branch B (CNN), then generates the final submission for the Test set.

    Args:
        lgb_oof_df (pd.DataFrame, optional): OOF predictions from LightGBM.
        lgb_test_df (pd.DataFrame, optional): Test predictions from LightGBM.
        cnn_oof_df (pd.DataFrame, optional): OOF predictions from CNN.
        cnn_test_df (pd.DataFrame, optional): Test predictions from CNN.

    If DataFrames are not provided, attempts to load them from the default paths
    defined in Config.WORKING_DIR.

    Returns:
        tuple: (submission_df, model)
    """
    # ==========================================
    # 1. Load Data if not provided
    # ==========================================
    if lgb_oof_df is None:
        path = os.path.join(Config.WORKING_DIR, "lgbm_oof.csv")
        if os.path.exists(path):
            logger.info(f"Loading LightGBM OOF from {path}")
            lgb_oof_df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"LightGBM OOF file not found at {path}")

    if lgb_test_df is None:
        path = os.path.join(Config.WORKING_DIR, "lgbm_test.csv")
        if os.path.exists(path):
            logger.info(f"Loading LightGBM Test from {path}")
            lgb_test_df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"LightGBM Test file not found at {path}")

    if cnn_oof_df is None:
        path = os.path.join(Config.WORKING_DIR, "cnn_oof.csv")
        if os.path.exists(path):
            logger.info(f"Loading CNN OOF from {path}")
            cnn_oof_df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"CNN OOF file not found at {path}")

    if cnn_test_df is None:
        path = os.path.join(Config.WORKING_DIR, "cnn_test.csv")
        if os.path.exists(path):
            logger.info(f"Loading CNN Test from {path}")
            cnn_test_df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"CNN Test file not found at {path}")

    # ==========================================
    # 2. Data Alignment & Merging
    # ==========================================
    logger.info("Aligning OOF predictions...")

    # Merge OOF DataFrames on segment_id
    # Rename columns to avoid collision
    lgb_oof_renamed = lgb_oof_df.rename(columns={"pred": "pred_lgb"})
    cnn_oof_renamed = cnn_oof_df.rename(columns={"pred": "pred_cnn"})

    # We assume both have 'time_to_eruption' and 'segment_id'
    # Drop target from one to avoid duplication during merge, but keep it for validation
    meta_train_df = pd.merge(
        lgb_oof_renamed[["segment_id", "time_to_eruption", "pred_lgb"]],
        cnn_oof_renamed[["segment_id", "pred_cnn"]],
        on="segment_id",
        how="inner",
    )

    if len(meta_train_df) != len(lgb_oof_df):
        logger.warning(
            f"Mismatch in OOF lengths after merge. Original: {len(lgb_oof_df)}, Merged: {len(meta_train_df)}"
        )

    # Prepare Training Data for Meta-Learner
    X_meta_train = meta_train_df[["pred_lgb", "pred_cnn"]].values
    y_meta_train = meta_train_df["time_to_eruption"].values

    logger.info("Aligning Test predictions...")
    lgb_test_renamed = lgb_test_df.rename(columns={"time_to_eruption": "pred_lgb"})
    cnn_test_renamed = cnn_test_df.rename(columns={"time_to_eruption": "pred_cnn"})

    meta_test_df = pd.merge(
        lgb_test_renamed[["segment_id", "pred_lgb"]],
        cnn_test_renamed[["segment_id", "pred_cnn"]],
        on="segment_id",
        how="inner",
    )

    X_meta_test = meta_test_df[["pred_lgb", "pred_cnn"]].values

    # ==========================================
    # 3. Train Ridge Meta-Learner
    # ==========================================
    logger.info("Training Ridge Meta-Learner...")

    model = Ridge(**Config.META_PARAMS)
    model.fit(X_meta_train, y_meta_train)

    # Evaluate on OOF (Proxy for CV score)
    oof_preds_meta = model.predict(X_meta_train)
    oof_mae = mean_absolute_error(y_meta_train, oof_preds_meta)

    logger.info(f"Meta-Learner OOF MAE: {oof_mae}")
    logger.info(
        f"Meta-Learner Coefficients: LGB={model.coef_[0]:.4f}, CNN={model.coef_[1]:.4f}"
    )
    logger.info(f"Meta-Learner Intercept: {model.intercept_:.4f}")

    # ==========================================
    # 4. Generate Submission
    # ==========================================
    logger.info("Generating Final Test Predictions...")

    final_test_preds = model.predict(X_meta_test)

    # Ensure non-negative predictions (time cannot be negative)
    final_test_preds = np.maximum(final_test_preds, 0)

    submission_df = pd.DataFrame(
        {
            "segment_id": meta_test_df["segment_id"],
            "time_to_eruption": final_test_preds,
        }
    )

    # Save Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Saved final submission to {submission_path}")
    logger.info(f"Submission shape: {submission_df.shape}")

    # Also save the meta-learner model for reference
    # (Optional, but good for reproducibility)
    # import joblib
    # joblib.dump(model, os.path.join(Config.WORKING_DIR, "meta_ridge_model.pkl"))

    return submission_df, model
