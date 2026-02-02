import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib
from library.config import STREAM_A_PARAMS, STREAM_B_PARAMS, NEG_POS_RATIO, WORKING_DIR
from library.utils import calc_mcc


class ContactXGB:
    def __init__(self):
        """
        Initializes the Dual-Stream XGBoost wrapper.
        Stream A: Interaction Model (Player-Player)
        Stream B: Impact Model (Player-Ground)
        """
        self.model_a = xgb.XGBClassifier(**STREAM_A_PARAMS)
        self.model_b = xgb.XGBClassifier(**STREAM_B_PARAMS)

        # Default thresholds, will be updated by optimize_thresholds
        self.threshold_a = 0.5
        self.threshold_b = 0.5

        self.models_ready = False

    def _undersample(self, X, y):
        """
        Applies Targeted Majority Undersampling.
        Retains 100% of positives, subsamples negatives to NEG_POS_RATIO.
        """
        if y is None:
            return X, y

        # Convert to numpy if pandas
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y

        pos_mask = y_arr == 1
        neg_mask = y_arr == 0

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        n_pos = len(pos_indices)
        n_neg_keep = int(n_pos * NEG_POS_RATIO)

        # If we have fewer negatives than the ratio implies, keep all negatives
        if n_neg_keep > len(neg_indices):
            n_neg_keep = len(neg_indices)

        # Randomly sample negatives
        # Seed is set globally in utils, but we use np.random here which respects it
        neg_indices_sampled = np.random.choice(neg_indices, n_neg_keep, replace=False)

        # Combine
        keep_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(keep_indices)

        if isinstance(X, pd.DataFrame):
            X_resampled = X.iloc[keep_indices].copy()
        else:
            X_resampled = X_arr[keep_indices]

        y_resampled = y_arr[keep_indices]

        return X_resampled, y_resampled

    def fit(self, train_data, val_data=None):
        """
        Trains both Stream A and Stream B models.

        Args:
            train_data (dict): {'stream_a': {'X': ..., 'y': ...}, 'stream_b': ...}
            val_data (dict, optional): Same structure as train_data for validation.
        """
        print("Starting training for ContactXGB...")

        # --- Stream A Training ---
        print("\n--- Training Stream A (Interaction Model) ---")
        X_train_a = train_data["stream_a"]["X"]
        y_train_a = train_data["stream_a"]["y"]

        # Apply Undersampling
        print(
            f"Original Stream A Train Shape: {X_train_a.shape}, Positives: {sum(y_train_a)}"
        )
        X_train_a_res, y_train_a_res = self._undersample(X_train_a, y_train_a)
        print(
            f"Resampled Stream A Train Shape: {X_train_a_res.shape}, Positives: {sum(y_train_a_res)}"
        )

        eval_set_a = None
        if val_data and "stream_a" in val_data:
            X_val_a = val_data["stream_a"]["X"]
            y_val_a = val_data["stream_a"]["y"]
            eval_set_a = [(X_train_a_res, y_train_a_res), (X_val_a, y_val_a)]

        self.model_a.fit(X_train_a_res, y_train_a_res, eval_set=eval_set_a, verbose=100)

        # --- Stream B Training ---
        print("\n--- Training Stream B (Impact Model) ---")
        X_train_b = train_data["stream_b"]["X"]
        y_train_b = train_data["stream_b"]["y"]

        # Apply Undersampling
        print(
            f"Original Stream B Train Shape: {X_train_b.shape}, Positives: {sum(y_train_b)}"
        )
        X_train_b_res, y_train_b_res = self._undersample(X_train_b, y_train_b)
        print(
            f"Resampled Stream B Train Shape: {X_train_b_res.shape}, Positives: {sum(y_train_b_res)}"
        )

        eval_set_b = None
        if val_data and "stream_b" in val_data:
            X_val_b = val_data["stream_b"]["X"]
            y_val_b = val_data["stream_b"]["y"]
            eval_set_b = [(X_train_b_res, y_train_b_res), (X_val_b, y_val_b)]

        self.model_b.fit(X_train_b_res, y_train_b_res, eval_set=eval_set_b, verbose=100)

        self.models_ready = True
        print("\nTraining completed.")

    def optimize_thresholds(self, val_data):
        """
        Finds optimal probability thresholds for MCC maximization on validation set.
        Updates self.threshold_a and self.threshold_b.
        """
        if not self.models_ready:
            raise RuntimeError("Models must be trained before optimizing thresholds.")

        print("\n--- Optimizing Thresholds ---")

        # Define search space
        thresholds = np.linspace(0.01, 0.99, 99)

        # --- Optimize Stream A ---
        if "stream_a" in val_data:
            X_val_a = val_data["stream_a"]["X"]
            y_val_a = val_data["stream_a"]["y"]

            if len(y_val_a) > 0:
                probas_a = self.model_a.predict_proba(X_val_a)[:, 1]
                best_mcc_a = -1.0
                best_thresh_a = 0.5

                for t in thresholds:
                    preds = (probas_a >= t).astype(int)
                    score = calc_mcc(y_val_a, preds)
                    if score > best_mcc_a:
                        best_mcc_a = score
                        best_thresh_a = t

                self.threshold_a = best_thresh_a
                print(
                    f"Stream A Optimal Threshold: {self.threshold_a:.4f} (MCC: {best_mcc_a})"
                )
            else:
                print("Stream A validation empty, keeping default threshold.")

        # --- Optimize Stream B ---
        if "stream_b" in val_data:
            X_val_b = val_data["stream_b"]["X"]
            y_val_b = val_data["stream_b"]["y"]

            if len(y_val_b) > 0:
                probas_b = self.model_b.predict_proba(X_val_b)[:, 1]
                best_mcc_b = -1.0
                best_thresh_b = 0.5

                for t in thresholds:
                    preds = (probas_b >= t).astype(int)
                    score = calc_mcc(y_val_b, preds)
                    if score > best_mcc_b:
                        best_mcc_b = score
                        best_thresh_b = t

                self.threshold_b = best_thresh_b
                print(
                    f"Stream B Optimal Threshold: {self.threshold_b:.4f} (MCC: {best_mcc_b})"
                )
            else:
                print("Stream B validation empty, keeping default threshold.")

    def predict(self, test_data):
        """
        Generates predictions for test data.

        Args:
            test_data (dict): {'stream_a': {'X': ..., 'ids': ...}, 'stream_b': ...}

        Returns:
            pd.DataFrame: DataFrame with columns ['contact_id', 'contact']
        """
        if not self.models_ready:
            raise RuntimeError("Models must be trained or loaded before prediction.")

        results = []

        # --- Predict Stream A ---
        if "stream_a" in test_data:
            X_test_a = test_data["stream_a"]["X"]
            ids_a = test_data["stream_a"]["ids"]

            if len(ids_a) > 0:
                probas_a = self.model_a.predict_proba(X_test_a)[:, 1]
                preds_a = (probas_a >= self.threshold_a).astype(int)

                df_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})
                results.append(df_a)

        # --- Predict Stream B ---
        if "stream_b" in test_data:
            X_test_b = test_data["stream_b"]["X"]
            ids_b = test_data["stream_b"]["ids"]

            if len(ids_b) > 0:
                probas_b = self.model_b.predict_proba(X_test_b)[:, 1]
                preds_b = (probas_b >= self.threshold_b).astype(int)

                df_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})
                results.append(df_b)

        if not results:
            return pd.DataFrame(columns=["contact_id", "contact"])

        # Combine results
        final_df = pd.concat(results, axis=0, ignore_index=True)
        return final_df

    def save(self, filename="best_model.joblib"):
        """Saves the entire wrapper object (models + thresholds)."""
        path = os.path.join(WORKING_DIR, filename)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    def load(self, filename="best_model.joblib"):
        """Loads the wrapper object."""
        path = os.path.join(WORKING_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        loaded_obj = joblib.load(path)
        self.model_a = loaded_obj.model_a
        self.model_b = loaded_obj.model_b
        self.threshold_a = loaded_obj.threshold_a
        self.threshold_b = loaded_obj.threshold_b
        self.models_ready = True
        print(f"Model loaded from {path}")
