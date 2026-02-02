import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from library.feature_engineering import create_dataset
from library.coordinate_utils import ECEF_to_WGS84, ENU_to_WGS84

# Constants
CACHE_DIR = "./working/idea_18/models"
SUBMISSION_DIR = "./submission"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Feature list based on gnss_physics.py output
FEATURES = [
    "Force_PR_E",
    "Force_PR_N",
    "Force_PR_U",
    "Force_Dop_E",
    "Force_Dop_N",
    "Force_Dop_U",
    "Cn0DbHz_mean",
    "Cn0DbHz_std",
    "Cn0DbHz_max",
    "Svid_count",
]


class LightGBMEnsemble:
    """
    Ensemble model wrapping two LightGBM regressors for East and North components.
    Predicts ENU residuals (GT - WLS).
    """

    def __init__(self, params=None):
        # Default parameters optimized for robustness (MAE)
        self.params = (
            params
            if params
            else {
                "objective": "mae",
                "n_estimators": 2000,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
        )
        self.model_e = None
        self.model_n = None

    def fit(
        self,
        X,
        y_e,
        y_n,
        X_val=None,
        y_e_val=None,
        y_n_val=None,
        early_stopping_rounds=50,
    ):
        """
        Trains the East and North models with early stopping.
        """
        # Train East Model
        self.model_e = lgb.LGBMRegressor(**self.params)
        eval_set_e = [(X_val, y_e_val)] if X_val is not None else None

        callbacks_e = []
        if early_stopping_rounds:
            callbacks_e.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )

        self.model_e.fit(
            X, y_e, eval_set=eval_set_e, eval_metric="mae", callbacks=callbacks_e
        )

        # Train North Model
        self.model_n = lgb.LGBMRegressor(**self.params)
        eval_set_n = [(X_val, y_n_val)] if X_val is not None else None

        callbacks_n = []
        if early_stopping_rounds:
            callbacks_n.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )

        self.model_n.fit(
            X, y_n, eval_set=eval_set_n, eval_metric="mae", callbacks=callbacks_n
        )

    def predict(self, X):
        """
        Predicts East and North residuals.
        """
        pred_e = self.model_e.predict(X)
        pred_n = self.model_n.predict(X)
        return pred_e, pred_n


def train_model(load_cached_data=True, n_folds=5, seed=42):
    """
    Trains the ensemble using GroupKFold cross-validation.

    Args:
        load_cached_data (bool): Whether to load dataset from cache.
        n_folds (int): Number of CV folds.
        seed (int): Random seed.

    Returns:
        list: List of trained LightGBMEnsemble objects (one per fold).
    """
    print("Loading training data...")
    df = create_dataset("train", load_cached_data=load_cached_data)

    # Drop rows where targets are NaN (e.g. if WLS failed or GT missing)
    df = df.dropna(subset=["target_E", "target_N"])

    # Extract data
    X = df[FEATURES]
    y_e = df["target_E"]
    y_n = df["target_N"]
    groups = df["drive_id"]

    gkf = GroupKFold(n_splits=n_folds)

    models = []
    oof_preds_e = np.zeros(len(df))
    oof_preds_n = np.zeros(len(df))

    print(f"Starting training with {n_folds} folds...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_e, groups)):
        X_train, y_e_train, y_n_train = (
            X.iloc[train_idx],
            y_e.iloc[train_idx],
            y_n.iloc[train_idx],
        )
        X_val, y_e_val, y_n_val = X.iloc[val_idx], y_e.iloc[val_idx], y_n.iloc[val_idx]

        model = LightGBMEnsemble()
        model.fit(
            X_train,
            y_e_train,
            y_n_train,
            X_val,
            y_e_val,
            y_n_val,
            early_stopping_rounds=50,
        )

        # Validation prediction
        p_e, p_n = model.predict(X_val)
        oof_preds_e[val_idx] = p_e
        oof_preds_n[val_idx] = p_n

        # Metrics for this fold
        mae_e = mean_absolute_error(y_e_val, p_e)
        mae_n = mean_absolute_error(y_n_val, p_n)
        print(
            f"Fold {fold+1} MAE - East: {mae_e:.10f}, North: {mae_n:.10f}, Avg: {(mae_e+mae_n)/2:.10f}"
        )

        models.append(model)

    # Overall Metrics
    total_mae_e = mean_absolute_error(y_e, oof_preds_e)
    total_mae_n = mean_absolute_error(y_n, oof_preds_n)
    print(
        f"\nOverall CV MAE - East: {total_mae_e:.10f}, North: {total_mae_n:.10f}, Avg: {(total_mae_e+total_mae_n)/2:.10f}"
    )

    return models


def generate_submission(models, load_cached_data=True):
    """
    Generates predictions for the test set and saves submission.csv.

    Args:
        models (list): List of trained LightGBMEnsemble models.
        load_cached_data (bool): Whether to load dataset from cache.
    """
    print("\nLoading test data...")
    df_test = create_dataset("test", load_cached_data=load_cached_data)

    # Features
    X_test = df_test[FEATURES]

    # Predict with all models and average (Bagging)
    preds_e = np.zeros(len(df_test))
    preds_n = np.zeros(len(df_test))

    print("Predicting on test set...")
    for model in models:
        pe, pn = model.predict(X_test)
        preds_e += pe
        preds_n += pn

    preds_e /= len(models)
    preds_n /= len(models)

    # Convert Predictions (ENU Residuals) back to Lat/Lon
    # 1. Get WLS Reference in LLA
    # We fill NaNs with 0 to allow the function to run, though rows with NaN WLS will likely be invalid.
    wls_x = df_test["WlsPositionXEcefMeters"].fillna(0).values
    wls_y = df_test["WlsPositionYEcefMeters"].fillna(0).values
    wls_z = df_test["WlsPositionZEcefMeters"].fillna(0).values

    ref_lat, ref_lon, ref_alt = ECEF_to_WGS84(wls_x, wls_y, wls_z)

    # 2. Apply ENU offsets to WLS reference to get final Lat/Lon
    # Target = GT - WLS  =>  GT = WLS + Target
    # We pass the predicted E, N (and U=0) as the "ENU coordinates" relative to the WLS reference.
    pred_lat, pred_lon, _ = ENU_to_WGS84(
        preds_e, preds_n, np.zeros_like(preds_e), ref_lat, ref_lon, ref_alt
    )

    # Create submission dataframe
    submission = pd.DataFrame(
        {
            "tripId": df_test["tripId"],
            "UnixTimeMillis": df_test["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
