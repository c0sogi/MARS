import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

from library.config import XGB_PARAMS, TRAIN_PARAMS, SUBMISSION_PATH


class EnergyPredictor:
    """
    Handles training and inference for energy prediction models using XGBoost.
    Predicts 'formation_energy_ev_natom' and 'bandgap_energy_ev'.
    """

    def __init__(self, xgb_params=None, train_params=None):
        """
        Initialize the predictor with XGBoost and training parameters.

        Args:
            xgb_params (dict): Hyperparameters for the XGBoost regressor.
            train_params (dict): Parameters for the training process (e.g., early_stopping).
        """
        self.xgb_params = xgb_params if xgb_params else XGB_PARAMS.copy()
        self.train_params = train_params if train_params else TRAIN_PARAMS.copy()
        self.models = {}
        self.targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
        self.feature_cols = None

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding metadata and target columns.
        """
        exclude_cols = ["id", "file_path"] + self.targets
        return [c for c in df.columns if c not in exclude_cols]

    def train(
        self, train_df, val_df, debug_sample_size=None, n_estimators_override=None
    ):
        """
        Trains separate XGBoost models for each target variable.

        Args:
            train_df (pd.DataFrame): Training data with features and targets.
            val_df (pd.DataFrame): Validation data with features and targets.
            debug_sample_size (int, optional): If set, limits training data size for debugging.
            n_estimators_override (int, optional): Overrides the number of estimators in XGB_PARAMS.
        """
        # Apply debugging subsample if requested
        if debug_sample_size and debug_sample_size < len(train_df):
            print(f"DEBUG: Subsampling training data to {debug_sample_size} samples.")
            train_df = train_df.sample(n=debug_sample_size, random_state=42)

        # Update params if override provided
        current_params = self.xgb_params.copy()
        if n_estimators_override:
            current_params["n_estimators"] = n_estimators_override

        # Determine feature columns from the dataframe
        self.feature_cols = self._get_feature_columns(train_df)
        print(f"Training with {len(self.feature_cols)} features.")

        X_train = train_df[self.feature_cols]
        X_val = val_df[self.feature_cols]

        for target in self.targets:
            print(f"\n--- Training for Target: {target} ---")

            y_train = train_df[target]
            y_val = val_df[target]

            # Log-transform targets: z = log(1 + y)
            # This helps with the distribution and ensures positivity predictions later
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)

            model = xgb.XGBRegressor(**current_params)

            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
                early_stopping_rounds=self.train_params.get(
                    "early_stopping_rounds", 50
                ),
                verbose=self.train_params.get("verbose", False),
            )

            # Evaluate on validation set
            val_preds_log = model.predict(X_val)
            val_rmsle = np.sqrt(mean_squared_error(y_val_log, val_preds_log))

            # Print full precision metric
            print(f"Validation RMSLE ({target}): {val_rmsle}")

            self.models[target] = model

    def predict(self, test_df):
        """
        Generates predictions for the test dataset.

        Args:
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            pd.DataFrame: DataFrame containing 'id' and predicted targets.
        """
        if not self.models:
            raise RuntimeError("Models have not been trained. Call train() first.")

        X_test = test_df[self.feature_cols]

        # Initialize results dataframe
        results = pd.DataFrame()
        results["id"] = test_df["id"]

        for target in self.targets:
            model = self.models[target]

            # Predict in log space
            pred_log = model.predict(X_test)

            # Inverse transform: y = exp(z) - 1
            pred = np.expm1(pred_log)

            # Ensure non-negative predictions (physical constraint)
            pred = np.maximum(pred, 0)

            results[target] = pred

        return results

    def save_submission(self, test_df, output_path=None):
        """
        Generates predictions and saves them to a CSV file.

        Args:
            test_df (pd.DataFrame): Test data.
            output_path (str, optional): Path to save the submission CSV.
                                         Defaults to configured SUBMISSION_PATH.
        """
        if output_path is None:
            output_path = SUBMISSION_PATH

        print(f"\nGenerating submission for {len(test_df)} samples...")
        submission_df = self.predict(test_df)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission_df
