import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from library.config import XGB_PARAMS, NEG_POS_RATIO, SEED, WORKING_DIR
from library.utils import compute_mcc, seed_everything


class DualStreamGBDT:
    """
    Implements the Asymmetric Dual-Stream GBDT model.
    Stream A: Interaction Model (Player-Player)
    Stream B: Impact Model (Player-Ground)
    """

    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.thresh_a = 0.5
        self.thresh_b = 0.5

        # Separate training params from model params
        self.xgb_params = XGB_PARAMS.copy()
        self.early_stopping_rounds = self.xgb_params.pop("early_stopping_rounds", 100)

        seed_everything(SEED)

    def _undersample(self, X, y):
        """
        Performs random undersampling of the majority class.
        """
        # Ensure inputs are pandas/numpy compatible
        # We keep X as DataFrame if it is one, to preserve feature names for XGBoost

        pos_mask = y == 1
        neg_mask = y == 0

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        n_pos = len(pos_indices)
        n_neg_keep = int(n_pos * NEG_POS_RATIO)

        if len(neg_indices) > n_neg_keep:
            selected_neg_indices = np.random.choice(
                neg_indices, n_neg_keep, replace=False
            )
        else:
            selected_neg_indices = neg_indices

        keep_indices = np.concatenate([pos_indices, selected_neg_indices])
        np.random.shuffle(keep_indices)

        if isinstance(X, pd.DataFrame):
            return X.iloc[keep_indices], y[keep_indices]

        return X[keep_indices], y[keep_indices]

    def _optimize_threshold(self, y_true, y_probs, stream_name):
        """
        Finds the probability threshold that maximizes MCC.
        """
        best_mcc = -1.0
        best_thresh = 0.5

        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            score = compute_mcc(y_true, preds)

            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh

        print(f"Stream {stream_name} Best Threshold: {best_thresh}")
        print(f"Stream {stream_name} Validation MCC: {best_mcc}")
        return best_thresh

    def _train_single_stream(self, X_train, y_train, X_val, y_val, stream_name):
        """
        Trains a single XGBoost stream with undersampling and early stopping.
        """
        print(f"\n--- Training Stream {stream_name} ---")
        print(
            f"Original Train Shape: {X_train.shape}, Positive Rate: {np.mean(y_train)}"
        )

        # Undersample Training Data
        X_train_res, y_train_res = self._undersample(X_train, y_train)
        print(
            f"Resampled Train Shape: {X_train_res.shape}, Positive Rate: {np.mean(y_train_res)}"
        )

        # Initialize Model
        params = self.xgb_params.copy()
        if "early_stopping_rounds" not in params:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        clf = xgb.XGBClassifier(**params)

        # Fit Model
        clf.fit(
            X_train_res,
            y_train_res,
            eval_set=[(X_train_res, y_train_res), (X_val, y_val)],
            verbose=100,
        )

        return clf

    def train(self, train_data, val_data):
        """
        Trains both streams and optimizes thresholds.

        Args:
            train_data (dict): Dictionary containing Stream A and B training data.
            val_data (dict): Dictionary containing Stream A and B validation data.
        """
        # --- Stream A ---
        self.model_a = self._train_single_stream(
            train_data["stream_a"]["X"],
            train_data["stream_a"]["y"],
            val_data["stream_a"]["X"],
            val_data["stream_a"]["y"],
            "A",
        )

        # Optimize Threshold A
        val_probs_a = self.model_a.predict_proba(val_data["stream_a"]["X"])[:, 1]
        self.thresh_a = self._optimize_threshold(
            val_data["stream_a"]["y"], val_probs_a, "A"
        )

        # --- Stream B ---
        self.model_b = self._train_single_stream(
            train_data["stream_b"]["X"],
            train_data["stream_b"]["y"],
            val_data["stream_b"]["X"],
            val_data["stream_b"]["y"],
            "B",
        )

        # Optimize Threshold B
        val_probs_b = self.model_b.predict_proba(val_data["stream_b"]["X"])[:, 1]
        self.thresh_b = self._optimize_threshold(
            val_data["stream_b"]["y"], val_probs_b, "B"
        )

    def predict(self, test_data):
        """
        Generates predictions for the test set using the dual-stream architecture.

        Args:
            test_data (dict): Dictionary containing Stream A and B test data.

        Returns:
            pd.DataFrame: DataFrame with 'contact_id' and 'contact' columns.
        """
        results = []

        # --- Predict Stream A ---
        if len(test_data["stream_a"]["ids"]) > 0:
            probs_a = self.model_a.predict_proba(test_data["stream_a"]["X"])[:, 1]
            preds_a = (probs_a >= self.thresh_a).astype(int)
            df_a = pd.DataFrame(
                {"contact_id": test_data["stream_a"]["ids"], "contact": preds_a}
            )
            results.append(df_a)

        # --- Predict Stream B ---
        if len(test_data["stream_b"]["ids"]) > 0:
            probs_b = self.model_b.predict_proba(test_data["stream_b"]["X"])[:, 1]
            preds_b = (probs_b >= self.thresh_b).astype(int)
            df_b = pd.DataFrame(
                {"contact_id": test_data["stream_b"]["ids"], "contact": preds_b}
            )
            results.append(df_b)

        # Combine
        if results:
            final_df = pd.concat(results, axis=0)
        else:
            final_df = pd.DataFrame(columns=["contact_id", "contact"])

        return final_df

    def save(self, filename="dual_stream_gbdt.joblib"):
        """Saves the model instance to disk."""
        path = os.path.join(WORKING_DIR, filename)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(filename="dual_stream_gbdt.joblib"):
        """Loads a model instance from disk."""
        path = os.path.join(WORKING_DIR, filename)
        if os.path.exists(path):
            print(f"Loading model from {path}")
            return joblib.load(path)
        else:
            print(f"No model found at {path}")
            return None


def generate_submission(model, test_data, output_path):
    """
    Generates predictions and saves the submission file.
    """
    print("Generating predictions for submission...")
    pred_df = model.predict(test_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    pred_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {pred_df.shape}")
