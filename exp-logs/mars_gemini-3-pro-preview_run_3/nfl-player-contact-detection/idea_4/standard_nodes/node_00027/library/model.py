import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
from library.config import XGB_PARAMS, WORKING_DIR, SUBMISSION_PATH
from library.utils import calc_mcc


class DualStreamXGB:
    """
    A wrapper class that manages two separate XGBoost models:
    - Stream A: For Player-Player contact detection.
    - Stream B: For Player-Ground contact detection.
    """

    def __init__(self):
        # Initialize models with parameters from config
        # We use a copy to ensure we don't mutate the global config if we were to change anything
        self.params = XGB_PARAMS.copy()

        self.model_a = xgb.XGBClassifier(**self.params)
        self.model_b = xgb.XGBClassifier(**self.params)

        # Initialize thresholds to default 0.5
        self.threshold_a = 0.5
        self.threshold_b = 0.5

    def fit(self, train_data, val_data):
        """
        Trains both Stream A and Stream B models using the provided training and validation data.

        Args:
            train_data (dict): Dictionary containing 'stream_a' and 'stream_b' training data.
            val_data (dict): Dictionary containing 'stream_a' and 'stream_b' validation data.
        """
        # ---------------------------------------------------------------------
        # Train Stream A (Player-Player)
        # ---------------------------------------------------------------------
        print("\n[DualStreamXGB] Training Stream A (Player-Player)...")
        X_train_a = train_data["stream_a"]["X"]
        y_train_a = train_data["stream_a"]["y"]
        X_val_a = val_data["stream_a"]["X"]
        y_val_a = val_data["stream_a"]["y"]

        # Fit with early stopping
        self.model_a.fit(
            X_train_a,
            y_train_a,
            eval_set=[(X_train_a, y_train_a), (X_val_a, y_val_a)],
            verbose=100,
        )

        # ---------------------------------------------------------------------
        # Train Stream B (Player-Ground)
        # ---------------------------------------------------------------------
        print("\n[DualStreamXGB] Training Stream B (Player-Ground)...")
        X_train_b = train_data["stream_b"]["X"]
        y_train_b = train_data["stream_b"]["y"]
        X_val_b = val_data["stream_b"]["X"]
        y_val_b = val_data["stream_b"]["y"]

        # Fit with early stopping
        self.model_b.fit(
            X_train_b,
            y_train_b,
            eval_set=[(X_train_b, y_train_b), (X_val_b, y_val_b)],
            verbose=100,
        )

    def optimize_thresholds(self, val_data):
        """
        Finds the optimal probability threshold for each stream independently
        to maximize the Matthews Correlation Coefficient (MCC) on the validation set.

        Args:
            val_data (dict): Dictionary containing 'stream_a' and 'stream_b' validation data.
        """
        print("\n[DualStreamXGB] Optimizing Thresholds...")

        # --- Optimize Stream A ---
        X_val_a = val_data["stream_a"]["X"]
        y_val_a = val_data["stream_a"]["y"]

        if len(X_val_a) > 0:
            # Get probabilities for the positive class
            probs_a = self.model_a.predict_proba(X_val_a)[:, 1]

            best_mcc_a = -1.0
            best_thresh_a = 0.5

            # Linear search for best threshold
            thresholds = np.linspace(0.01, 0.99, 99)

            for t in thresholds:
                preds = (probs_a >= t).astype(int)
                score = calc_mcc(y_val_a, preds)
                if score > best_mcc_a:
                    best_mcc_a = score
                    best_thresh_a = t

            self.threshold_a = best_thresh_a
            print(f"Stream A - Best Threshold: {best_thresh_a}, MCC: {best_mcc_a}")
        else:
            print("Stream A - No validation data found, skipping optimization.")

        # --- Optimize Stream B ---
        X_val_b = val_data["stream_b"]["X"]
        y_val_b = val_data["stream_b"]["y"]

        if len(X_val_b) > 0:
            probs_b = self.model_b.predict_proba(X_val_b)[:, 1]

            best_mcc_b = -1.0
            best_thresh_b = 0.5

            thresholds = np.linspace(0.01, 0.99, 99)

            for t in thresholds:
                preds = (probs_b >= t).astype(int)
                score = calc_mcc(y_val_b, preds)
                if score > best_mcc_b:
                    best_mcc_b = score
                    best_thresh_b = t

            self.threshold_b = best_thresh_b
            print(f"Stream B - Best Threshold: {best_thresh_b}, MCC: {best_mcc_b}")
        else:
            print("Stream B - No validation data found, skipping optimization.")

        # --- Calculate Global MCC ---
        if len(X_val_a) > 0 and len(X_val_b) > 0:
            # Re-predict using optimal thresholds
            preds_a = (probs_a >= self.threshold_a).astype(int)
            preds_b = (probs_b >= self.threshold_b).astype(int)

            # Concatenate all predictions and labels
            all_preds = np.concatenate([preds_a, preds_b])
            all_true = np.concatenate([y_val_a, y_val_b])

            global_mcc = calc_mcc(all_true, all_preds)
            print(f"Global Validation MCC: {global_mcc}")

    def predict(self, test_data):
        """
        Generates predictions for the test set using the trained models and optimized thresholds.

        Args:
            test_data (dict): Dictionary containing 'stream_a' and 'stream_b' test data.

        Returns:
            pd.DataFrame: DataFrame containing 'contact_id' and 'contact' predictions.
        """
        print("\n[DualStreamXGB] Generating Predictions...")

        # --- Predict Stream A ---
        X_test_a = test_data["stream_a"]["X"]
        ids_a = test_data["stream_a"]["ids"]

        if len(X_test_a) > 0:
            probs_a = self.model_a.predict_proba(X_test_a)[:, 1]
            preds_a = (probs_a >= self.threshold_a).astype(int)
            df_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})
        else:
            df_a = pd.DataFrame(columns=["contact_id", "contact"])

        # --- Predict Stream B ---
        X_test_b = test_data["stream_b"]["X"]
        ids_b = test_data["stream_b"]["ids"]

        if len(X_test_b) > 0:
            probs_b = self.model_b.predict_proba(X_test_b)[:, 1]
            preds_b = (probs_b >= self.threshold_b).astype(int)
            df_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})
        else:
            df_b = pd.DataFrame(columns=["contact_id", "contact"])

        # --- Combine ---
        df_sub = pd.concat([df_a, df_b], ignore_index=True)
        return df_sub

    def save_models(self):
        """
        Saves the trained models (JSON) and thresholds (JSON) to the working directory.
        """
        path_a = os.path.join(WORKING_DIR, "model_a.json")
        path_b = os.path.join(WORKING_DIR, "model_b.json")
        path_t = os.path.join(WORKING_DIR, "thresholds.json")

        self.model_a.save_model(path_a)
        self.model_b.save_model(path_b)

        thresholds = {
            "threshold_a": float(self.threshold_a),
            "threshold_b": float(self.threshold_b),
        }

        with open(path_t, "w") as f:
            json.dump(thresholds, f)

        print(f"Models and thresholds saved to {WORKING_DIR}")

    def generate_submission(self, test_data):
        """
        Generates predictions for the test set and saves the submission CSV file.

        Args:
            test_data (dict): Dictionary containing 'stream_a' and 'stream_b' test data.
        """
        df_preds = self.predict(test_data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        df_preds.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Submission shape: {df_preds.shape}")
