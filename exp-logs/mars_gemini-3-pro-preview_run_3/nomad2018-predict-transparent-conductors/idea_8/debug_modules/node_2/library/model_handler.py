import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, SUBMISSION_DIR
from library.feature_engineering import build_feature_matrix


class EnergyPredictor:
    def __init__(self):
        """
        Initializes the EnergyPredictor with XGBoost regressors for both targets.
        """
        self.model_formation = xgb.XGBRegressor(**XGB_PARAMS)
        self.model_bandgap = xgb.XGBRegressor(**XGB_PARAMS)
        self.targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    def _transform_target(self, y):
        """
        Applies log(1 + y) transformation to the target.
        """
        return np.log1p(y)

    def _inverse_transform_target(self, z):
        """
        Applies exp(z) - 1 transformation to the prediction.
        """
        return np.expm1(z)

    def train(self, train_metadata, val_metadata, load_cached_data=True):
        """
        Trains the models on the provided metadata.

        Args:
            train_metadata (pd.DataFrame): Training metadata.
            val_metadata (pd.DataFrame): Validation metadata.
            load_cached_data (bool): Whether to use cached feature matrices.
        """
        print("Preparing feature matrices for training...")
        # Build feature matrices
        X_train = build_feature_matrix(
            train_metadata, "train", load_cached_data=load_cached_data
        )
        X_val = build_feature_matrix(
            val_metadata, "val", load_cached_data=load_cached_data
        )

        # Ensure alignment just in case, though build_feature_matrix handles index
        # We assume the metadata index aligns with the feature matrix index

        for target in self.targets:
            print(f"\nTraining model for target: {target}")

            # Prepare targets with log transformation
            y_train = self._transform_target(train_metadata[target])
            y_val = self._transform_target(val_metadata[target])

            model = (
                self.model_formation
                if target == "formation_energy_ev_natom"
                else self.model_bandgap
            )

            # Train with early stopping
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False,
            )

            # Evaluation
            # Predict on validation set (log space)
            z_pred_val = model.predict(X_val)

            # Calculate MSE in log space (which is RMSLE in original space squared)
            mse_log = mean_squared_error(y_val, z_pred_val)
            rmsle = np.sqrt(mse_log)

            print(f"Validation RMSLE for {target}: {rmsle}")

            # Also print RMSE in original space for reference
            y_pred_val = self._inverse_transform_target(z_pred_val)
            y_true_val = val_metadata[target].values
            rmse_orig = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
            print(f"Validation RMSE (original scale) for {target}: {rmse_orig}")

    def predict(self, test_metadata, load_cached_data=True):
        """
        Generates predictions for the test set.

        Args:
            test_metadata (pd.DataFrame): Test metadata.
            load_cached_data (bool): Whether to use cached feature matrices.

        Returns:
            pd.DataFrame: DataFrame containing 'id' and predictions.
        """
        print("\nPreparing feature matrix for inference...")
        X_test = build_feature_matrix(
            test_metadata, "test", load_cached_data=load_cached_data
        )

        predictions = pd.DataFrame({"id": test_metadata["id"]})

        for target in self.targets:
            print(f"Predicting {target}...")
            model = (
                self.model_formation
                if target == "formation_energy_ev_natom"
                else self.model_bandgap
            )

            # Predict in log space
            z_pred = model.predict(X_test)

            # Inverse transform to original space
            y_pred = self._inverse_transform_target(z_pred)

            # Ensure no negative predictions (physical constraint)
            y_pred = np.maximum(y_pred, 0)

            predictions[target] = y_pred

        return predictions

    def generate_submission(self, test_metadata, output_path, load_cached_data=True):
        """
        Generates predictions and saves them to a CSV file.

        Args:
            test_metadata (pd.DataFrame): Test metadata.
            output_path (str): Path to save the submission CSV.
            load_cached_data (bool): Whether to use cached feature matrices.
        """
        preds_df = self.predict(test_metadata, load_cached_data=load_cached_data)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        preds_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
