import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS
from library.preprocessing import inverse_log_transform


class EnergyPredictor:
    def __init__(self):
        """
        Initializes the EnergyPredictor with storage for models.
        """
        self.models = {}
        self.targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    def train(self, X_train, y_train, X_val, y_val, verbose=False):
        """
        Trains separate XGBoost models for each target variable using early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Log-transformed training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Log-transformed validation targets.
            verbose (bool): Whether to print detailed training logs from XGBoost.
        """
        print("Starting training of XGBoost models...")

        for target in self.targets:
            print(f"\nTraining model for target: {target}")

            # Initialize model with hyperparameters from config
            # Fix: Move early_stopping_rounds to constructor for XGBoost >= 1.6 compatibility
            params = XGB_PARAMS.copy()
            params["early_stopping_rounds"] = 100
            model = XGBRegressor(**params)

            # Fit the model
            # Note: The targets passed here are already log-transformed by the preprocessing pipeline.
            # We use the validation set for early stopping to prevent overfitting.
            model.fit(
                X_train,
                y_train[target],
                eval_set=[(X_val, y_val[target])],
                verbose=verbose,
            )

            # Store the trained model
            self.models[target] = model

            # Evaluate on validation set
            # Predictions are in log space (log(1+y))
            val_preds_log = model.predict(X_val)

            # Metric Calculation:
            # The competition metric is Column-wise Root Mean Squared Logarithmic Error (RMSLE).
            # Since our target y is log1p(original_y), RMSE on y corresponds directly to RMSLE on original_y.
            mse_log = mean_squared_error(y_val[target], val_preds_log)
            rmsle = np.sqrt(mse_log)

            print(f"Validation RMSLE for {target}: {rmsle}")

            # Calculate MAE in original scale for physical interpretability
            val_preds_orig = inverse_log_transform(val_preds_log)
            y_val_orig = inverse_log_transform(y_val[target])
            mae_orig = np.mean(np.abs(val_preds_orig - y_val_orig))
            print(f"Validation MAE (Original Scale) for {target}: {mae_orig}")

    def predict(self, X_test):
        """
        Generates predictions for the test set using the trained models.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: DataFrame containing 'id' and predicted targets in original scale.
        """
        print("\nGenerating predictions for test set...")

        if not self.models:
            raise RuntimeError("Models have not been trained yet. Call train() first.")

        predictions = {}

        for target in self.targets:
            model = self.models[target]

            # Predict (output is in log space)
            log_preds = model.predict(X_test)

            # Transform back to original scale (exp(y) - 1)
            orig_preds = inverse_log_transform(log_preds)

            predictions[target] = orig_preds

        # Create submission DataFrame
        # Assuming X_test index corresponds to the 'id'
        submission_df = pd.DataFrame(predictions, index=X_test.index)

        # Reset index to make 'id' a column
        submission_df.reset_index(inplace=True)

        # Ensure correct column order: id, formation_energy_ev_natom, bandgap_energy_ev
        cols = ["id"] + self.targets
        submission_df = submission_df[cols]

        return submission_df
