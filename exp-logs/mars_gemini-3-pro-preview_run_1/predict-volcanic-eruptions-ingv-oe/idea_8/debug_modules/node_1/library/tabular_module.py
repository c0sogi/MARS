import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library import config, utils

# Ensure reproducibility
utils.seed_everything(config.SEED)


def train_tabular_model(X_train, y_train, X_val, y_val, X_test=None, fold_idx=0):
    """
    Trains a LightGBM model for a single fold using the configuration from library.config.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.array): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (np.array): Validation targets.
        X_test (pd.DataFrame, optional): Test features for prediction.
        fold_idx (int): Index of the current fold (for logging and saving).

    Returns:
        model (lgb.LGBMRegressor): The trained model wrapper.
        val_preds (np.array): Predictions on the validation set (OOF).
        test_preds (np.array): Predictions on the test set (if X_test is provided).
    """
    # 1. Prepare Configuration
    # Copy params to avoid modifying the global config
    params = config.LGBM_PARAMS.copy()

    # Extract parameters that shouldn't be passed to the constructor or need specific handling
    early_stopping_round = params.pop("early_stopping_round", 100)
    verbose_eval = params.pop("verbose", -1)

    # 2. Data Cleaning
    # Remove metadata columns like 'segment_id' that are not features
    drop_cols = ["segment_id"]

    X_train_clean = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val_clean = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])

    if X_test is not None:
        X_test_clean = X_test.drop(
            columns=[c for c in drop_cols if c in X_test.columns]
        )
    else:
        X_test_clean = None

    # 3. Initialize Model
    # We use the sklearn API for consistency, passing remaining params as kwargs
    model = lgb.LGBMRegressor(**params)

    # 4. Define Callbacks for Early Stopping and Logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_round, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    print(f"Training LightGBM Tabular Model for Fold {fold_idx}...")

    # 5. Train
    model.fit(
        X_train_clean,
        y_train,
        eval_set=[(X_train_clean, y_train), (X_val_clean, y_val)],
        eval_names=["train", "valid"],
        eval_metric="mae",
        callbacks=callbacks,
    )

    # 6. Generate Predictions
    # Predict on validation set (OOF)
    val_preds = model.predict(X_val_clean)
    # Enforce physical constraint: time >= 0
    val_preds = np.maximum(0, val_preds)

    # Predict on test set if provided
    test_preds = None
    if X_test_clean is not None:
        test_preds = model.predict(X_test_clean)
        test_preds = np.maximum(0, test_preds)

    # 7. Evaluation & Saving
    mae = utils.mae_score(y_val, val_preds)
    utils.print_metric(f"Fold {fold_idx} Tabular MAE", mae)

    # Save the underlying booster to a text file
    model_save_path = os.path.join(config.LGBM_MODEL_DIR, f"lgbm_fold_{fold_idx}.txt")
    model.booster_.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, val_preds, test_preds


def predict_tabular_model(X_test, fold_idx):
    """
    Loads a trained LightGBM model for a specific fold and generates predictions.

    Args:
        X_test (pd.DataFrame): Test features.
        fold_idx (int): The fold index corresponding to the model to load.

    Returns:
        np.array: Predictions for the test set.
    """
    model_path = os.path.join(config.LGBM_MODEL_DIR, f"lgbm_fold_{fold_idx}.txt")

    if not os.path.exists(model_path):
        print(
            f"Warning: Model for fold {fold_idx} not found at {model_path}. Returning zeros."
        )
        return np.zeros(len(X_test))

    # Load the model
    booster = lgb.Booster(model_file=model_path)

    # Clean data
    drop_cols = ["segment_id"]
    X_test_clean = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # Predict
    preds = booster.predict(X_test_clean)

    # Enforce physical constraint
    preds = np.maximum(0, preds)

    return preds
