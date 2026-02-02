import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.config import LGBM_PARAMS, EARLY_STOPPING_ROUNDS, SEED, WORKING_DIR

# Define features used for training
# These correspond to the columns generated in library.feature_eng.py
FEATURES = [
    # L1 Pseudorange Forces
    "F_L1_E",
    "F_L1_N",
    "F_L1_U",
    "W_L1",
    # L5 Pseudorange Forces
    "F_L5_E",
    "F_L5_N",
    "F_L5_U",
    "W_L5",
    # Doppler Forces
    "F_Dop_E",
    "F_Dop_N",
    "F_Dop_U",
    "W_Dop",
    # Signal Quality / Context
    "Cn0DbHz",
    "Svid",
    "L5_Count",
]


class LGBMResidualPredictor:
    """
    Wrapper for LightGBM models to predict ENU residuals from geometric features.
    Trains separate models for Easting and Northing errors.
    """

    def __init__(self):
        self.models_e = []
        self.models_n = []
        self.feature_cols = FEATURES
        self.params = LGBM_PARAMS.copy()

    def train(self, train_df, val_df=None):
        """
        Train the residual predictors using GroupKFold cross-validation.

        Args:
            train_df (pd.DataFrame): Training data containing features and targets.
            val_df (pd.DataFrame, optional): Validation data. If provided, it's used
                                             for evaluation but K-Fold is done on train_df.
        """
        # Ensure targets exist
        if "Target_E" not in train_df.columns or "Target_N" not in train_df.columns:
            raise ValueError(
                "Target columns (Target_E, Target_N) missing from training data."
            )

        # Prepare GroupKFold
        # We group by drive_id to ensure generalization to new drives
        groups = train_df["drive_id"]
        gkf = GroupKFold(n_splits=5)

        print(
            f"Training LightGBM Residual Predictor with {len(self.feature_cols)} features..."
        )

        # --- Train Easting Model ---
        print("\n--- Training Easting (E) Model ---")
        self.models_e = self._train_target(train_df, "Target_E", gkf, groups)

        # --- Train Northing Model ---
        print("\n--- Training Northing (N) Model ---")
        self.models_n = self._train_target(train_df, "Target_N", gkf, groups)

        # --- Evaluation on Validation Set (if provided) ---
        if val_df is not None:
            print("\n--- Evaluating on Hold-out Validation Set ---")
            preds = self.predict(val_df)

            # Calculate MAE
            mae_e = np.mean(np.abs(val_df["Target_E"] - preds["pred_E"]))
            mae_n = np.mean(np.abs(val_df["Target_N"] - preds["pred_N"]))

            print(f"Validation MAE East:  {mae_e:.9f} m")
            print(f"Validation MAE North: {mae_n:.9f} m")

            # Calculate composite metric (mean of component MAEs)
            print(f"Validation Mean Component MAE: {(mae_e + mae_n) / 2:.9f} m")

    def _train_target(self, df, target_col, folder, groups):
        """
        Helper to train models for a specific target column using CV.
        """
        models = []
        oof_preds = np.zeros(len(df))

        for fold, (train_idx, valid_idx) in enumerate(
            folder.split(df, df[target_col], groups)
        ):
            X_train = df.iloc[train_idx][self.feature_cols]
            y_train = df.iloc[train_idx][target_col]

            X_valid = df.iloc[valid_idx][self.feature_cols]
            y_valid = df.iloc[valid_idx][target_col]

            # Create LGBM Datasets
            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

            # Train
            model = lgb.train(
                self.params,
                dtrain,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                num_boost_round=self.params["n_estimators"],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(period=0),  # Suppress per-iteration logging
                ],
            )

            # Predict
            val_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            oof_preds[valid_idx] = val_pred

            # Score
            score = np.mean(np.abs(y_valid - val_pred))
            print(f"Fold {fold} MAE: {score:.9f}")

            models.append(model)

        total_mae = np.mean(np.abs(df[target_col] - oof_preds))
        print(f"Overall CV MAE for {target_col}: {total_mae:.9f}")

        return models

    def predict(self, test_df):
        """
        Generate predictions using the trained ensemble.

        Args:
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            pd.DataFrame: DataFrame with 'UnixTimeMillis', 'pred_E', 'pred_N'.
        """
        X_test = test_df[self.feature_cols]

        # East Predictions
        pred_e = np.zeros(len(test_df))
        for model in self.models_e:
            pred_e += model.predict(X_test, num_iteration=model.best_iteration)
        if self.models_e:
            pred_e /= len(self.models_e)

        # North Predictions
        pred_n = np.zeros(len(test_df))
        for model in self.models_n:
            pred_n += model.predict(X_test, num_iteration=model.best_iteration)
        if self.models_n:
            pred_n /= len(self.models_n)

        result = pd.DataFrame(
            {
                "UnixTimeMillis": test_df["UnixTimeMillis"],
                "pred_E": pred_e,
                "pred_N": pred_n,
            }
        )

        return result
