import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from library.config import (
    LGB_PARAMS,
    N_FOLDS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SUBMISSION_DIR,
    METADATA_DIR,
)
from library.data_loader import get_dataset


def train_model(train_df):
    """
    Trains LightGBM ensemble models for East and North residuals using GroupKFold.

    Args:
        train_df (pd.DataFrame): Training data containing features and targets.

    Returns:
        tuple: (models_E, models_N, feature_cols)
            models_E: List of trained LightGBM models for East residual.
            models_N: List of trained LightGBM models for North residual.
            feature_cols: List of feature column names used for training.
    """
    # Identify feature columns based on prefixes defined in feature engineering
    feature_cols = [
        c
        for c in train_df.columns
        if c.startswith(("pr_", "doppler_", "Cn0", "SvEl", "global_", "imu_"))
    ]
    print(f"Training with {len(feature_cols)} features.")

    X = train_df[feature_cols]
    y_E = train_df["target_E"]
    y_N = train_df["target_N"]
    groups = train_df["drive_id"]

    models_E = []
    models_N = []

    gkf = GroupKFold(n_splits=N_FOLDS)

    # --- Train East Component ---
    print("\nTraining East Component...")
    oof_preds_E = np.zeros(len(train_df))

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y_E, groups)):
        X_train, y_train = X.iloc[trn_idx], y_E.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y_E.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            LGB_PARAMS,
            lgb_train,
            valid_sets=[lgb_eval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(VERBOSE_EVAL),
            ],
        )
        models_E.append(model)
        oof_preds_E[val_idx] = model.predict(X_val)

    mae_E = mean_absolute_error(y_E, oof_preds_E)
    print(f"East Component CV MAE: {mae_E}")

    # --- Train North Component ---
    print("\nTraining North Component...")
    oof_preds_N = np.zeros(len(train_df))

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y_N, groups)):
        X_train, y_train = X.iloc[trn_idx], y_N.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y_N.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            LGB_PARAMS,
            lgb_train,
            valid_sets=[lgb_eval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(VERBOSE_EVAL),
            ],
        )
        models_N.append(model)
        oof_preds_N[val_idx] = model.predict(X_val)

    mae_N = mean_absolute_error(y_N, oof_preds_N)
    print(f"North Component CV MAE: {mae_N}")

    return models_E, models_N, feature_cols


def predict(models, df, feature_cols):
    """
    Generates predictions using an ensemble of models and aggregates via median.

    Args:
        models (list): List of trained LightGBM models.
        df (pd.DataFrame): Dataframe containing features.
        feature_cols (list): List of feature column names.

    Returns:
        np.array: Aggregated predictions.
    """
    X = df[feature_cols]
    preds = []
    for model in models:
        preds.append(model.predict(X))

    # Pixel-wise median for robustness against outliers
    return np.median(np.column_stack(preds), axis=1)


def generate_submission(load_cached_data=True):
    """
    End-to-end pipeline: Load data, Train, Validate, Predict, and Save Submission.
    """
    # 1. Load Data
    print("Loading datasets...")
    train_df = get_dataset(
        os.path.join(METADATA_DIR, "train_metadata.csv"), load_cached_data, "train"
    )
    val_df = get_dataset(
        os.path.join(METADATA_DIR, "val_metadata.csv"), load_cached_data, "val"
    )
    test_df = get_dataset(
        os.path.join(METADATA_DIR, "test_metadata.csv"), load_cached_data, "test"
    )

    # 2. Train Models
    models_E, models_N, feature_cols = train_model(train_df)

    # 3. Evaluation on Hold-out Validation Set
    print("\nEvaluating on Hold-out Validation Set...")
    val_pred_E = predict(models_E, val_df, feature_cols)
    val_pred_N = predict(models_N, val_df, feature_cols)

    mae_val_E = mean_absolute_error(val_df["target_E"], val_pred_E)
    mae_val_N = mean_absolute_error(val_df["target_N"], val_pred_N)

    print(f"Validation MAE East: {mae_val_E}")
    print(f"Validation MAE North: {mae_val_N}")
    # Approximation of 2D error
    mean_dist_error = (mae_val_E + mae_val_N) / 2 * np.sqrt(2)
    print(f"Estimated Mean Distance Error: {mean_dist_error}")

    # 4. Inference on Test Set
    print("\nGenerating Test Predictions...")
    test_pred_E = predict(models_E, test_df, feature_cols)
    test_pred_N = predict(models_N, test_df, feature_cols)

    # 5. Reconstruct Trajectory (WLS + Predicted Residuals)
    # Convert meters back to degrees
    # dLat = dN / 111320
    # dLon = dE / (111320 * cos(lat))

    # Use WLS lat for cosine projection to avoid circular dependency, accurate enough for small residuals
    wls_lat_rad = np.radians(test_df["wls_lat"])

    pred_lat = test_df["wls_lat"] + (test_pred_N / 111320.0)
    pred_lon = test_df["wls_lon"] + (test_pred_E / (111320.0 * np.cos(wls_lat_rad)))

    # 6. Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": test_df["tripId"],
            "UnixTimeMillis": test_df.index,
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Ensure directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
