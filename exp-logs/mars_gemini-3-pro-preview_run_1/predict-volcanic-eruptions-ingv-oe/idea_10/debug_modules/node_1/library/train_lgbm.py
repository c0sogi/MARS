import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib

from library.config import Config
from library.dataset import VolcanoTabularBuilder
from library.utils import get_logger, seed_everything

# Initialize logger
logger = get_logger("train_lgbm")


def run_lgbm_cv(debug=False):
    """
    Executes the training pipeline for Branch A: Robust Audio-Seismic Regressor (LightGBM).

    Performs 5-Fold Cross-Validation to:
    1. Train LightGBM models on robust tabular features.
    2. Generate Out-of-Fold (OOF) predictions for the Meta-Learner.
    3. Generate averaged predictions for the Test set.

    Args:
        debug (bool): If True, runs on a smaller subset of data for rapid testing.

    Returns:
        tuple: (oof_df, test_pred_df)
    """
    seed_everything(Config.SEED)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    builder = VolcanoTabularBuilder()

    # Load Train and Val sets separately, then combine for full K-Fold CV
    # We load cached data if available to save time
    df_train_part, _, _ = builder.get_data("train", load_cache=True)
    df_val_part, _, _ = builder.get_data("val", load_cache=True)

    # Concatenate to create the full training set for CV
    df_full_train = pd.concat([df_train_part, df_val_part], axis=0).reset_index(
        drop=True
    )

    # Load Test set
    df_test, _, _ = builder.get_data("test", load_cache=True)

    if debug:
        logger.info("DEBUG mode enabled: Subsampling data.")
        df_full_train = df_full_train.sample(
            n=min(100, len(df_full_train)), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(50, len(df_test)), random_state=Config.SEED
        ).reset_index(drop=True)

    # Define Features and Target
    # Exclude non-feature columns
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in df_full_train.columns if c not in exclude_cols]
    target_col = "time_to_eruption"

    logger.info(
        f"Training on {len(df_full_train)} samples with {len(feature_cols)} features."
    )

    # ==========================================
    # 2. Cross-Validation Setup
    # ==========================================
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Storage for OOF and Test predictions
    oof_preds = np.zeros(len(df_full_train))
    test_preds = np.zeros(len(df_test))

    # Prepare LightGBM Parameters
    params = Config.LGB_PARAMS.copy()

    # Extract training-specific args that shouldn't be in the params dict for lgb.train
    num_boost_round = params.pop("n_estimators", 5000)
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)
    verbose_eval = 100

    if debug:
        num_boost_round = 50
        verbose_eval = 10

    # ==========================================
    # 3. Training Loop
    # ==========================================
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_full_train)):
        logger.info(f"\n--- Starting Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split Data
        X_train = df_full_train.iloc[train_idx][feature_cols]
        y_train = df_full_train.iloc[train_idx][target_col]
        X_val = df_full_train.iloc[val_idx][feature_cols]
        y_val = df_full_train.iloc[val_idx][target_col]

        # Create LightGBM Datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Callbacks for Early Stopping and Logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=verbose_eval),
        ]

        # Train Model
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Predict on Validation (OOF)
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        # Predict on Test
        # Accumulate predictions (will average later)
        test_pred_fold = model.predict(
            df_test[feature_cols], num_iteration=model.best_iteration
        )
        test_preds += test_pred_fold / Config.N_FOLDS

        # Evaluate Fold
        fold_mae = mean_absolute_error(y_val, val_pred)
        fold_scores.append(fold_mae)
        logger.info(f"Fold {fold + 1} MAE: {fold_mae}")

        # Save Model
        model_path = os.path.join(Config.WORKING_DIR, f"lgb_model_fold_{fold}.txt")
        model.save_model(model_path)

    # ==========================================
    # 4. Results & Saving
    # ==========================================
    overall_mae = mean_absolute_error(df_full_train[target_col], oof_preds)
    logger.info(f"\nOverall CV MAE: {overall_mae}")
    logger.info(f"Average Fold MAE: {np.mean(fold_scores)}")

    # Construct DataFrames for Output
    oof_df = df_full_train[["segment_id", target_col]].copy()
    oof_df["pred"] = oof_preds

    test_pred_df = df_test[["segment_id"]].copy()
    test_pred_df["time_to_eruption"] = test_preds

    # Save OOF and Test predictions for Stacking (Meta-Learner)
    oof_save_path = os.path.join(Config.WORKING_DIR, "lgbm_oof.csv")
    test_save_path = os.path.join(Config.WORKING_DIR, "lgbm_test.csv")

    oof_df.to_csv(oof_save_path, index=False)
    test_pred_df.to_csv(test_save_path, index=False)

    logger.info(f"Saved OOF predictions to {oof_save_path}")
    logger.info(f"Saved Test predictions to {test_save_path}")

    # Save Final Submission (Standalone LightGBM result)
    # This ensures a valid submission exists even if the stacking step fails or is skipped
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    test_pred_df.to_csv(submission_path, index=False)
    logger.info(f"Saved submission file to {submission_path}")

    return oof_df, test_pred_df
