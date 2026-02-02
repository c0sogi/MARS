import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from library.config import Config
from library.utils import seed_everything


class DualStreamGBDT:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)

        self.model_a = None
        self.model_b = None

    def _undersample(self, X: pd.DataFrame, y: np.ndarray):
        """
        Performs Targeted Majority Undersampling.
        Retains 100% of positive class.
        Subsamples negative class to achieve a specific Negative:Positive ratio.

        Config.NEGATIVE_SAMPLE_RATIO = 0.1 implies Positive / Negative = 0.1
        Therefore, n_negatives = n_positives / 0.1 = n_positives * 10.
        """
        # Ensure y is a 1D array
        y = y.ravel()

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        if n_pos == 0:
            print(
                "Warning: No positive samples found in training data. Skipping undersampling."
            )
            return X, y

        # Calculate target number of negatives
        # Ratio = Pos / Neg  => Neg = Pos / Ratio
        target_n_neg = int(n_pos / self.config.NEGATIVE_SAMPLE_RATIO)

        if target_n_neg < n_neg:
            # Randomly sample negatives
            rng = np.random.RandomState(self.config.SEED)
            keep_neg_indices = rng.choice(neg_indices, size=target_n_neg, replace=False)
        else:
            # Keep all negatives if we don't have enough to meet the ratio
            keep_neg_indices = neg_indices

        # Combine indices
        keep_indices = np.concatenate([pos_indices, keep_neg_indices])

        # Shuffle to mix classes
        np.random.shuffle(keep_indices)

        # Subset data
        X_resampled = X.iloc[keep_indices].copy()
        y_resampled = y[keep_indices]

        return X_resampled, y_resampled

    def train(self, data_train: dict, data_val: dict):
        """
        Trains both Stream A and Stream B models.

        Args:
            data_train (dict): {'stream_a': (X, y, ids), 'stream_b': (X, y, ids)}
            data_val (dict): {'stream_a': (X, y, ids), 'stream_b': (X, y, ids)}
        """
        print("\n=== Training DualStreamGBDT ===")

        # --- Train Stream A (Interaction) ---
        print("Training Stream A (Interaction Model)...")
        X_a_train, y_a_train, _ = data_train["stream_a"]
        X_a_val, y_a_val, _ = data_val["stream_a"]

        if len(X_a_train) > 0:
            # Undersample Training Data
            print(
                f"  Original Stream A Train Shape: {X_a_train.shape}, Positives: {sum(y_a_train)}"
            )
            X_a_train_res, y_a_train_res = self._undersample(X_a_train, y_a_train)
            print(
                f"  Resampled Stream A Train Shape: {X_a_train_res.shape}, Positives: {sum(y_a_train_res)}"
            )

            self.model_a = xgb.XGBClassifier(**self.config.XGB_PARAMS_STREAM_A)

            self.model_a.fit(
                X_a_train_res,
                y_a_train_res,
                eval_set=[(X_a_train_res, y_a_train_res), (X_a_val, y_a_val)],
                verbose=False,
            )

            best_score = self.model_a.best_score
            print(f"  Stream A Best LogLoss: {best_score}")
        else:
            print("  Warning: Stream A training data is empty.")
            self.model_a = None

        # --- Train Stream B (Impact) ---
        print("\nTraining Stream B (Impact Model)...")
        X_b_train, y_b_train, _ = data_train["stream_b"]
        X_b_val, y_b_val, _ = data_val["stream_b"]

        if len(X_b_train) > 0:
            # Undersample Training Data
            print(
                f"  Original Stream B Train Shape: {X_b_train.shape}, Positives: {sum(y_b_train)}"
            )
            X_b_train_res, y_b_train_res = self._undersample(X_b_train, y_b_train)
            print(
                f"  Resampled Stream B Train Shape: {X_b_train_res.shape}, Positives: {sum(y_b_train_res)}"
            )

            self.model_b = xgb.XGBClassifier(**self.config.XGB_PARAMS_STREAM_B)

            self.model_b.fit(
                X_b_train_res,
                y_b_train_res,
                eval_set=[(X_b_train_res, y_b_train_res), (X_b_val, y_b_val)],
                verbose=False,
            )

            best_score = self.model_b.best_score
            print(f"  Stream B Best LogLoss: {best_score}")
        else:
            print("  Warning: Stream B training data is empty.")
            self.model_b = None

    def predict_proba(self, data_test: dict):
        """
        Generates probability predictions for both streams.

        Args:
            data_test (dict): {'stream_a': (X, y, ids), 'stream_b': (X, y, ids)}

        Returns:
            dict: {'stream_a': probas_array, 'stream_b': probas_array}
        """
        preds = {}

        # --- Predict Stream A ---
        X_a_test, _, _ = data_test["stream_a"]
        if self.model_a is not None and len(X_a_test) > 0:
            # Predict proba returns [prob_0, prob_1], we want prob_1
            preds["stream_a"] = self.model_a.predict_proba(X_a_test)[:, 1]
        else:
            preds["stream_a"] = np.zeros(len(X_a_test))

        # --- Predict Stream B ---
        X_b_test, _, _ = data_test["stream_b"]
        if self.model_b is not None and len(X_b_test) > 0:
            preds["stream_b"] = self.model_b.predict_proba(X_b_test)[:, 1]
        else:
            preds["stream_b"] = np.zeros(len(X_b_test))

        return preds

    def save(self, directory: str):
        """Saves the models to the specified directory."""
        os.makedirs(directory, exist_ok=True)
        path_a = os.path.join(directory, "model_stream_a.joblib")
        path_b = os.path.join(directory, "model_stream_b.joblib")

        if self.model_a:
            joblib.dump(self.model_a, path_a)
        if self.model_b:
            joblib.dump(self.model_b, path_b)
        print(f"Models saved to {directory}")

    def load(self, directory: str):
        """Loads the models from the specified directory."""
        path_a = os.path.join(directory, "model_stream_a.joblib")
        path_b = os.path.join(directory, "model_stream_b.joblib")

        if os.path.exists(path_a):
            self.model_a = joblib.load(path_a)
        else:
            print(f"Warning: Model A not found at {path_a}")

        if os.path.exists(path_b):
            self.model_b = joblib.load(path_b)
        else:
            print(f"Warning: Model B not found at {path_b}")
