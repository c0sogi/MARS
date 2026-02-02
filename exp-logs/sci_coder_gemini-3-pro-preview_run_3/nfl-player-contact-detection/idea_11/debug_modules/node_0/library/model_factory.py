import xgboost as xgb
import numpy as np
import pandas as pd
import os
import gc
from library.config import Config
from library.utils import compute_mcc, seed_everything


class DualStreamModel:
    """
    Implements the Robust Asymmetric Modality-Selective GBDT architecture.
    Manages two separate XGBoost models:
    1. Stream A: Interaction Model (Player-Player) - Uses Tracking + Conditional Visuals
    2. Stream B: Impact Model (Player-Ground) - Uses Tracking + Physics Derivatives (No Visuals)
    """

    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.threshold_a = 0.5
        self.threshold_b = 0.5
        self.config = Config

        # Ensure working directory exists for model artifacts
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def _undersample(self, X, y):
        """
        Performs Random Undersampling on the majority class (0).
        Keeps all positive samples (1).
        Downsamples negatives to maintain NEG_POS_RATIO.
        """
        # Ensure inputs are numpy arrays or pandas objects
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate number of negatives to keep
        n_neg_keep = int(n_pos * self.config.NEG_POS_RATIO)
        n_neg_keep = min(
            n_neg, n_neg_keep
        )  # Ensure we don't sample more than available

        # Randomly select negatives
        np.random.seed(self.config.SEED)
        neg_indices_selected = np.random.choice(
            neg_indices, size=n_neg_keep, replace=False
        )

        # Combine and shuffle
        all_indices = np.concatenate([pos_indices, neg_indices_selected])
        np.random.shuffle(all_indices)

        return X[all_indices], y[all_indices]

    def _train_stream(self, stream_name, X_train, y_train, X_val, y_val):
        """
        Generic training logic for a single stream.
        """
        print(f"--- Training {stream_name} ---")
        print(
            f"Original Train Shape: {X_train.shape}, Positive Rate: {np.mean(y_train)}"
        )

        # Apply Undersampling
        X_train_res, y_train_res = self._undersample(X_train, y_train)
        print(
            f"Resampled Train Shape: {X_train_res.shape}, Positive Rate: {np.mean(y_train_res)}"
        )

        # Configure XGBoost
        # We use the parameters from Config, ensuring 'missing' is set correctly
        params = self.config.XGB_PARAMS.copy()

        clf = xgb.XGBClassifier(**params)

        # Train with Early Stopping
        clf.fit(
            X_train_res,
            y_train_res,
            eval_set=[(X_val, y_val)],
            verbose=False,  # Silent execution as requested
        )

        # Evaluate
        best_iteration = clf.best_iteration
        best_score = clf.best_score
        print(f"{stream_name} Best Iteration: {best_iteration}")
        print(f"{stream_name} Best LogLoss: {best_score}")

        return clf

    def train(self, data_bundle):
        """
        Orchestrates training for both streams.

        Args:
            data_bundle (dict): Contains train/val data for both streams.
                Keys: X_train_A, y_train_A, X_val_A, y_val_A,
                      X_train_B, y_train_B, X_val_B, y_val_B
        """
        seed_everything(self.config.SEED)

        # --- Train Stream A ---
        self.model_a = self._train_stream(
            "Stream A (Interaction)",
            data_bundle["X_train_A"],
            data_bundle["y_train_A"],
            data_bundle["X_val_A"],
            data_bundle["y_val_A"],
        )

        # --- Train Stream B ---
        self.model_b = self._train_stream(
            "Stream B (Impact)",
            data_bundle["X_train_B"],
            data_bundle["y_train_B"],
            data_bundle["X_val_B"],
            data_bundle["y_val_B"],
        )

        # Save models immediately after training
        self.save_models()

    def optimize_thresholds(self, X_val_A, y_val_A, X_val_B, y_val_B):
        """
        Finds the optimal probability threshold for each stream to maximize MCC.
        """
        print("--- Optimizing Thresholds ---")

        def find_best_thresh(model, X, y, name):
            probas = model.predict_proba(X)[:, 1]

            best_mcc = -1.0
            best_t = 0.5

            # Search range defined in Config
            thresholds = np.arange(
                self.config.THRESHOLD_SEARCH_START,
                self.config.THRESHOLD_SEARCH_END,
                self.config.THRESHOLD_SEARCH_STEP,
            )

            for t in thresholds:
                preds = (probas >= t).astype(int)
                score = compute_mcc(y, preds)
                if score > best_mcc:
                    best_mcc = score
                    best_t = t

            print(f"{name} Optimal Threshold: {best_t}")
            print(f"{name} Max MCC: {best_mcc}")
            return best_t

        self.threshold_a = find_best_thresh(self.model_a, X_val_A, y_val_A, "Stream A")
        self.threshold_b = find_best_thresh(self.model_b, X_val_B, y_val_B, "Stream B")

    def predict(self, X_test_A, ids_test_A, X_test_B, ids_test_B):
        """
        Generates predictions for the test set.
        Routes data through appropriate models and applies optimized thresholds.

        Returns:
            pd.DataFrame: Submission dataframe with 'contact_id' and 'contact'.
        """
        results = []

        # --- Predict Stream A ---
        if X_test_A is not None and len(X_test_A) > 0:
            probas_a = self.model_a.predict_proba(X_test_A)[:, 1]
            preds_a = (probas_a >= self.threshold_a).astype(int)

            df_a = pd.DataFrame({"contact_id": ids_test_A, "contact": preds_a})
            results.append(df_a)

        # --- Predict Stream B ---
        if X_test_B is not None and len(X_test_B) > 0:
            probas_b = self.model_b.predict_proba(X_test_B)[:, 1]
            preds_b = (probas_b >= self.threshold_b).astype(int)

            df_b = pd.DataFrame({"contact_id": ids_test_B, "contact": preds_b})
            results.append(df_b)

        # Combine
        if not results:
            return pd.DataFrame(columns=["contact_id", "contact"])

        submission = pd.concat(results, axis=0, ignore_index=True)
        return submission

    def save_models(self):
        """
        Saves the trained XGBoost models to JSON.
        """
        path_a = os.path.join(self.config.WORKING_DIR, "model_streamA.json")
        path_b = os.path.join(self.config.WORKING_DIR, "model_streamB.json")

        if self.model_a:
            self.model_a.save_model(path_a)
        if self.model_b:
            self.model_b.save_model(path_b)

        # Save thresholds as well (simple text or json)
        thresh_path = os.path.join(self.config.WORKING_DIR, "thresholds.json")
        import json

        with open(thresh_path, "w") as f:
            json.dump(
                {"threshold_a": self.threshold_a, "threshold_b": self.threshold_b}, f
            )

        print(f"Models saved to {self.config.WORKING_DIR}")

    def load_models(self):
        """
        Loads models and thresholds from disk.
        """
        path_a = os.path.join(self.config.WORKING_DIR, "model_streamA.json")
        path_b = os.path.join(self.config.WORKING_DIR, "model_streamB.json")
        thresh_path = os.path.join(self.config.WORKING_DIR, "thresholds.json")

        if os.path.exists(path_a) and os.path.exists(path_b):
            # Re-initialize classifiers
            self.model_a = xgb.XGBClassifier(**self.config.XGB_PARAMS)
            self.model_a.load_model(path_a)

            self.model_b = xgb.XGBClassifier(**self.config.XGB_PARAMS)
            self.model_b.load_model(path_b)

            if os.path.exists(thresh_path):
                import json

                with open(thresh_path, "r") as f:
                    data = json.load(f)
                    self.threshold_a = data.get("threshold_a", 0.5)
                    self.threshold_b = data.get("threshold_b", 0.5)

            print("Models and thresholds loaded successfully.")
            return True
        else:
            print("Saved models not found.")
            return False
