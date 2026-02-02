import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, SUBMISSION_PATH


class DualModelWrapper:
    """
    Wrapper for training and predicting with two separate XGBoost models
    for Formation Energy and Bandgap Energy.
    """

    def __init__(self):
        # Separate fit params from constructor params
        self.xgb_params = XGB_PARAMS.copy()
        # Extract early_stopping_rounds if present, default to 50
        self.early_stopping_rounds = self.xgb_params.pop("early_stopping_rounds", 50)

        # Initialize models
        self.model_formation = xgb.XGBRegressor(**self.xgb_params)
        self.model_bandgap = xgb.XGBRegressor(**self.xgb_params)

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains both models using the provided training and validation data.
        Expects y_train and y_val to be log-transformed (log1p).
        """
        print("\n--- Training Formation Energy Model ---")
        self.model_formation.fit(
            X_train,
            y_train["formation_energy_log"],
            eval_set=[
                (X_train, y_train["formation_energy_log"]),
                (X_val, y_val["formation_energy_log"]),
            ],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=100,
        )

        print("\n--- Training Bandgap Energy Model ---")
        self.model_bandgap.fit(
            X_train,
            y_train["bandgap_energy_log"],
            eval_set=[
                (X_train, y_train["bandgap_energy_log"]),
                (X_val, y_val["bandgap_energy_log"]),
            ],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=100,
        )

        # Evaluate on validation set
        self.evaluate(X_val, y_val)

    def evaluate(self, X_val, y_val):
        """
        Evaluates the models on the validation set and prints RMSLE scores.
        Since inputs are log-transformed, RMSE on these equals RMSLE on original scale.
        """
        print("\n--- Validation Evaluation ---")

        # Predict in log space
        pred_form_log = self.model_formation.predict(X_val)
        pred_band_log = self.model_bandgap.predict(X_val)

        # Calculate RMSE (which is RMSLE in original space)
        rmse_form = np.sqrt(
            mean_squared_error(y_val["formation_energy_log"], pred_form_log)
        )
        rmse_band = np.sqrt(
            mean_squared_error(y_val["bandgap_energy_log"], pred_band_log)
        )

        print(f"Formation Energy RMSLE: {rmse_form:.10f}")
        print(f"Bandgap Energy RMSLE:   {rmse_band:.10f}")
        print(f"Mean Column-wise RMSLE: {(rmse_form + rmse_band) / 2:.10f}")

    def predict(self, X_test):
        """
        Generates predictions for the test set.
        Returns a dictionary with original scale predictions.
        """
        # Predict log values
        pred_form_log = self.model_formation.predict(X_test)
        pred_band_log = self.model_bandgap.predict(X_test)

        # Inverse transform (expm1) to get eV
        pred_form = np.expm1(pred_form_log)
        pred_band = np.expm1(pred_band_log)

        # Ensure non-negative (physics constraint)
        pred_form = np.maximum(pred_form, 0)
        pred_band = np.maximum(pred_band, 0)

        return {"formation_energy_ev_natom": pred_form, "bandgap_energy_ev": pred_band}


def generate_submission(model, X_test, ids, output_path=SUBMISSION_PATH):
    """
    Generates the submission CSV file using the trained model.
    """
    print(f"\nGenerating predictions for {len(X_test)} test samples...")
    preds = model.predict(X_test)

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds["formation_energy_ev_natom"],
            "bandgap_energy_ev": preds["bandgap_energy_ev"],
        }
    )

    # Ensure correct column order
    submission_df = submission_df[
        ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    ]

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())
