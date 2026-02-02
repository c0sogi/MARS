import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

from library.config import (
    XGB_PARAMS,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_CSV,
    RANDOM_SEED,
)
from library.preprocessor import TargetTransformer, get_preprocessed_data


class EnergyModel:
    def __init__(self, n_estimators=None, learning_rate=None, max_depth=None):
        """
        Initialize the EnergyModel with XGBoost regressors.
        Allows overriding default hyperparameters for debugging or tuning.
        """
        self.params = XGB_PARAMS.copy()
        if n_estimators is not None:
            self.params["n_estimators"] = n_estimators
        if learning_rate is not None:
            self.params["learning_rate"] = learning_rate
        if max_depth is not None:
            self.params["max_depth"] = max_depth

        # separate models for each target
        self.model_formation = xgb.XGBRegressor(**self.params)
        self.model_bandgap = xgb.XGBRegressor(**self.params)

        # transformers for log(1+y) scaling
        self.tf_formation = TargetTransformer()
        self.tf_bandgap = TargetTransformer()

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100):
        """
        Fit the models for formation energy and bandgap energy.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets ['formation_energy_ev_natom', 'bandgap_energy_ev'].
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.DataFrame, optional): Validation targets.
            early_stopping_rounds (int): Rounds for early stopping.
        """
        # Transform targets to log space
        y_train_form = self.tf_formation.transform(y_train["formation_energy_ev_natom"])
        y_train_band = self.tf_bandgap.transform(y_train["bandgap_energy_ev"])

        eval_set_form = None
        eval_set_band = None

        if X_val is not None and y_val is not None:
            y_val_form = self.tf_formation.transform(y_val["formation_energy_ev_natom"])
            y_val_band = self.tf_bandgap.transform(y_val["bandgap_energy_ev"])
            eval_set_form = [(X_train, y_train_form), (X_val, y_val_form)]
            eval_set_band = [(X_train, y_train_band), (X_val, y_val_band)]

        print("\nTraining Formation Energy Model...")
        self.model_formation.fit(
            X_train,
            y_train_form,
            eval_set=eval_set_form,
            early_stopping_rounds=early_stopping_rounds,
            verbose=100,
        )

        print("\nTraining Bandgap Energy Model...")
        self.model_bandgap.fit(
            X_train,
            y_train_band,
            eval_set=eval_set_band,
            early_stopping_rounds=early_stopping_rounds,
            verbose=100,
        )

        # Calculate and print validation metrics
        if X_val is not None and y_val is not None:
            print("\n--- Validation Metrics (RMSLE) ---")
            # Note: RMSE on log-transformed data is equivalent to RMSLE on original data

            # Formation Energy
            pred_val_form_trans = self.model_formation.predict(X_val)
            rmse_form = np.sqrt(mean_squared_error(y_val_form, pred_val_form_trans))
            print(f"Formation Energy RMSLE: {rmse_form}")

            # Bandgap Energy
            pred_val_band_trans = self.model_bandgap.predict(X_val)
            rmse_band = np.sqrt(mean_squared_error(y_val_band, pred_val_band_trans))
            print(f"Bandgap Energy RMSLE:   {rmse_band}")

            avg_rmsle = (rmse_form + rmse_band) / 2
            print(f"Average RMSLE:          {avg_rmsle}")

    def predict(self, X):
        """
        Predicts formation energy and bandgap energy for input features X.
        Applies inverse transformation to return values in original scale.

        Returns:
            pd.DataFrame: Predictions with columns ['formation_energy_ev_natom', 'bandgap_energy_ev']
        """
        pred_form_trans = self.model_formation.predict(X)
        pred_band_trans = self.model_bandgap.predict(X)

        # Inverse transform to get original scale
        pred_form = self.tf_formation.inverse_transform(pred_form_trans)
        pred_band = self.tf_bandgap.inverse_transform(pred_band_trans)

        # Ensure non-negative predictions physically
        pred_form = np.maximum(0, pred_form)
        pred_band = np.maximum(0, pred_band)

        return pd.DataFrame(
            {"formation_energy_ev_natom": pred_form, "bandgap_energy_ev": pred_band},
            index=X.index,
        )


def train_and_generate_submission(debug=False, n_estimators=None):
    """
    Orchestrates the full pipeline:
    1. Load and clean Train/Val data.
    2. Train XGBoost models.
    3. Load and clean Test data (using training cleaner).
    4. Generate predictions.
    5. Save submission file.

    Args:
        debug (bool): If True, uses a small subset of data for quick testing.
        n_estimators (int): Override number of estimators for the model.
    """
    # 1. Load and Preprocess Training Data
    print("Loading Training Data...")
    train_df, cleaner = get_preprocessed_data(split="train", load_cached_data=True)

    # 2. Load and Preprocess Validation Data
    print("Loading Validation Data...")
    val_df, _ = get_preprocessed_data(
        split="val", cleaner=cleaner, load_cached_data=True
    )

    if debug:
        print("Debug mode: Subsampling data.")
        train_df = train_df.head(500)
        val_df = val_df.head(100)
        if n_estimators is None:
            n_estimators = 100

    # Prepare Feature Matrices and Target Vectors
    # Exclude non-feature columns
    exclude_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "file_path"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]

    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]

    print(f"Training with {len(feature_cols)} features.")

    # 3. Train Model
    model = EnergyModel(n_estimators=n_estimators)
    model.fit(X_train, y_train, X_val, y_val)

    # 4. Load and Preprocess Test Data
    print("\nLoading Test Data...")
    test_df, _ = get_preprocessed_data(
        split="test", cleaner=cleaner, load_cached_data=True
    )

    # Ensure test set has same features as training
    # (The cleaner ensures columns dropped in train are dropped in test,
    # but we must ensure we select the same remaining columns)
    X_test = test_df[feature_cols]

    # 5. Generate Predictions
    print("Generating Predictions...")
    predictions = model.predict(X_test)

    # 6. Create Submission File
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "formation_energy_ev_natom": predictions["formation_energy_ev_natom"],
            "bandgap_energy_ev": predictions["bandgap_energy_ev"],
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission.head())
