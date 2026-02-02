import xgboost as xgb
import numpy as np
import pandas as pd
import os
import json
import gc
from library.config import Config
from library.utils import compute_mcc, set_seed


class DualStreamGBDT:
    """
    Implements the Biomechanical Dual-Stream GBDT model.
    Manages two separate XGBoost classifiers:
    - Stream A: Player-Player Interaction
    - Stream B: Player-Ground Impact
    """

    def __init__(self):
        self.config = Config
        self.model_a = None
        self.model_b = None
        self.threshold_a = 0.5
        self.threshold_b = 0.5

        # Persistence paths
        self.model_a_path = os.path.join(self.config.WORKING_DIR, "model_stream_a.json")
        self.model_b_path = os.path.join(self.config.WORKING_DIR, "model_stream_b.json")
        self.meta_path = os.path.join(self.config.WORKING_DIR, "model_metadata.json")

    def undersample(self, X, y):
        """
        Performs random undersampling of the negative class to achieve the configured ratio.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.array): Target labels.

        Returns:
            tuple: (X_resampled, y_resampled)
        """
        set_seed(self.config.SEED)

        # Identify indices
        pos_indices = np.flatnonzero(y == 1)
        neg_indices = np.flatnonzero(y == 0)

        n_pos = len(pos_indices)
        n_neg_target = int(n_pos * self.config.NEG_POS_RATIO)

        # Sample negatives
        if len(neg_indices) > n_neg_target:
            sampled_neg_indices = np.random.choice(
                neg_indices, n_neg_target, replace=False
            )
        else:
            sampled_neg_indices = neg_indices

        # Combine and shuffle
        indices = np.concatenate([pos_indices, sampled_neg_indices])
        np.random.shuffle(indices)

        return X.iloc[indices].copy(), y[indices].copy()

    def train_stream(self, X_train, y_train, X_val, y_val, params, stream_name):
        """
        Trains a single XGBoost stream with early stopping and validation.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            params (dict): XGBoost hyperparameters.
            stream_name (str): Name for logging.

        Returns:
            xgb.XGBClassifier: Trained model.
        """
        print(f"\n--- Training {stream_name} ---")
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # Initialize model
        # We pass early_stopping_rounds to fit(), but params to init
        model = xgb.XGBClassifier(**params)

        # Fit model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Log best score
        best_score = model.best_score
        print(f"{stream_name} Best LogLoss: {best_score}")

        return model

    def optimize_threshold(self, model, X_val, y_val, stream_name):
        """
        Finds the probability threshold that maximizes MCC on the validation set.

        Args:
            model: Trained XGBClassifier.
            X_val, y_val: Validation data.
            stream_name (str): Name for logging.

        Returns:
            tuple: (best_threshold, best_mcc)
        """
        print(f"Optimizing threshold for {stream_name}...")

        # Get probabilities for positive class
        probs = model.predict_proba(X_val)[:, 1]

        best_mcc = -1.0
        best_thresh = 0.5

        # Linear search
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            score = compute_mcc(y_val, preds)

            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh

        print(f"{stream_name} Optimal Threshold: {best_thresh}")
        print(f"{stream_name} Validation MCC: {best_mcc}")

        return best_thresh, best_mcc

    def train(self, train_data_a, val_data_a, train_data_b, val_data_b):
        """
        Main training pipeline for both streams.

        Args:
            train_data_a: (X, y, ids) for Stream A Train
            val_data_a: (X, y, ids) for Stream A Val
            train_data_b: (X, y, ids) for Stream B Train
            val_data_b: (X, y, ids) for Stream B Val
        """
        # Unpack data
        X_a_train, y_a_train, _ = train_data_a
        X_a_val, y_a_val, _ = val_data_a

        X_b_train, y_b_train, _ = train_data_b
        X_b_val, y_b_val, _ = val_data_b

        # --- Stream A ---
        # 1. Undersample
        X_a_res, y_a_res = self.undersample(X_a_train, y_a_train)

        # 2. Train
        self.model_a = self.train_stream(
            X_a_res, y_a_res, X_a_val, y_a_val, self.config.STREAM_A_PARAMS, "Stream A"
        )

        # 3. Optimize
        self.threshold_a, mcc_a = self.optimize_threshold(
            self.model_a, X_a_val, y_a_val, "Stream A"
        )

        # --- Stream B ---
        # 1. Undersample
        X_b_res, y_b_res = self.undersample(X_b_train, y_b_train)

        # 2. Train
        self.model_b = self.train_stream(
            X_b_res, y_b_res, X_b_val, y_b_val, self.config.STREAM_B_PARAMS, "Stream B"
        )

        # 3. Optimize
        self.threshold_b, mcc_b = self.optimize_threshold(
            self.model_b, X_b_val, y_b_val, "Stream B"
        )

        # Save models
        self.save_models()

        # Clean up
        gc.collect()

    def save_models(self):
        """Saves models and metadata to disk."""
        if self.model_a:
            self.model_a.save_model(self.model_a_path)
        if self.model_b:
            self.model_b.save_model(self.model_b_path)

        metadata = {
            "threshold_a": float(self.threshold_a),
            "threshold_b": float(self.threshold_b),
        }
        with open(self.meta_path, "w") as f:
            json.dump(metadata, f)

        print(f"Models saved to {self.config.WORKING_DIR}")

    def load_models(self):
        """Loads models and metadata from disk."""
        if os.path.exists(self.model_a_path):
            self.model_a = xgb.XGBClassifier()
            self.model_a.load_model(self.model_a_path)

        if os.path.exists(self.model_b_path):
            self.model_b = xgb.XGBClassifier()
            self.model_b.load_model(self.model_b_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                metadata = json.load(f)
                self.threshold_a = metadata.get("threshold_a", 0.5)
                self.threshold_b = metadata.get("threshold_b", 0.5)

        print("Models loaded successfully.")

    def generate_submission(self, test_data_a, test_data_b):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            test_data_a: (X, y, ids) for Stream A
            test_data_b: (X, y, ids) for Stream B
        """
        print("Generating submission...")

        results = []

        # --- Stream A Predictions ---
        if self.model_a and test_data_a[0].shape[0] > 0:
            X_a, _, ids_a = test_data_a
            probs_a = self.model_a.predict_proba(X_a)[:, 1]
            preds_a = (probs_a >= self.threshold_a).astype(int)

            df_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})
            results.append(df_a)

        # --- Stream B Predictions ---
        if self.model_b and test_data_b[0].shape[0] > 0:
            X_b, _, ids_b = test_data_b
            probs_b = self.model_b.predict_proba(X_b)[:, 1]
            preds_b = (probs_b >= self.threshold_b).astype(int)

            df_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})
            results.append(df_b)

        # Combine
        if results:
            df_pred = pd.concat(results, ignore_index=True)
        else:
            # Fallback for empty test set (unlikely)
            df_pred = pd.DataFrame(columns=["contact_id", "contact"])

        # Load sample submission to ensure all IDs are present and order is correct
        df_sample = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions into sample submission
        # We use left join on sample submission to keep its structure
        # Fill missing with 0 (no contact) as a safe default
        df_final = df_sample[["contact_id"]].merge(df_pred, on="contact_id", how="left")
        df_final["contact"] = df_final["contact"].fillna(0).astype(int)

        # Save
        df_final.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_final.shape}")
