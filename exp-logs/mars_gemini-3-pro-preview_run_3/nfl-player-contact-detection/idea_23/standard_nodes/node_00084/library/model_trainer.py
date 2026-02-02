import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib
from library.config import Config
from library.utils import seed_everything, compute_mcc


class ModelTrainer:
    def __init__(self):
        """
        Initializes the ModelTrainer.
        Sets the random seed and ensures the working directory exists.
        """
        seed_everything(Config.SEED)
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        self.models = {}
        self.thresholds = {}

    def _apply_undersampling(self, X, y):
        """
        Applies targeted majority undersampling.
        Keeps 100% of positive samples and subsamples negative samples
        to achieve the ratio defined in Config.UNDERSAMPLE_RATIO.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.array): Target vector.

        Returns:
            tuple: (X_resampled, y_resampled)
        """
        # Ensure y is a numpy array
        y = np.array(y)

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate number of negatives to keep
        n_neg_keep = int(n_pos * Config.UNDERSAMPLE_RATIO)
        n_neg_keep = min(n_neg_keep, n_neg)  # Ensure we don't take more than available

        # Randomly sample negatives
        np.random.seed(Config.SEED)
        neg_indices_sampled = np.random.choice(
            neg_indices, size=n_neg_keep, replace=False
        )

        # Combine and shuffle
        all_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(all_indices)

        X_resampled = X.iloc[all_indices].copy()
        y_resampled = y[all_indices]

        print(f"Undersampling: Pos={n_pos}, Neg={n_neg} -> Neg_Sampled={n_neg_keep}")

        return X_resampled, y_resampled

    def _optimize_threshold(self, y_true, y_probs):
        """
        Performs a linear search to find the probability threshold that maximizes MCC.

        Args:
            y_true (np.array): Ground truth labels.
            y_probs (np.array): Predicted probabilities.

        Returns:
            tuple: (best_threshold, best_mcc)
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search range from 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            score = compute_mcc(y_true, y_pred)

            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh

        return best_thresh, best_mcc

    def train_stream(self, X_train, y_train, X_val, y_val, stream_type):
        """
        Trains an XGBoost model for a specific stream (A or B).

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            stream_type (str): 'A' for Interaction, 'B' for Impact.

        Returns:
            tuple: (model, best_threshold, best_val_mcc)
        """
        print(f"\n=== Training Stream {stream_type} ===")

        # 1. Undersampling
        X_train_sub, y_train_sub = self._apply_undersampling(X_train, y_train)

        # 2. Configuration
        if stream_type == "A":
            params = Config.XGB_PARAMS_STREAM_A
        elif stream_type == "B":
            params = Config.XGB_PARAMS_STREAM_B
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")

        # 3. Initialize Model
        clf = xgb.XGBClassifier(
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS, **params
        )

        # 4. Train with Early Stopping
        # Note: early_stopping_rounds is passed to fit() in newer sklearn API,
        # but for compatibility with various versions we use the standard kwargs.
        clf.fit(
            X_train_sub,
            y_train_sub,
            eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
            verbose=Config.VERBOSE_EVAL,
        )

        # 5. Validation & Threshold Optimization
        print(f"Optimizing threshold for Stream {stream_type}...")
        val_probs = clf.predict_proba(X_val)[:, 1]
        best_thresh, best_mcc = self._optimize_threshold(y_val, val_probs)

        print(f"Stream {stream_type} Results:")
        print(f"Best Threshold: {best_thresh:.4f}")
        print(f"Validation MCC: {best_mcc}")  # Printing full precision as requested

        # 6. Save Model and Metadata
        self.models[stream_type] = clf
        self.thresholds[stream_type] = best_thresh

        model_path = os.path.join(self.working_dir, f"model_stream_{stream_type}.json")
        clf.save_model(model_path)

        return clf, best_thresh, best_mcc

    def predict_and_submit(self, ids_test_a, X_test_a, ids_test_b, X_test_b):
        """
        Generates predictions for the test set and creates the submission file.

        Args:
            ids_test_a (np.array): Contact IDs for Stream A.
            X_test_a (pd.DataFrame): Features for Stream A.
            ids_test_b (np.array): Contact IDs for Stream B.
            X_test_b (pd.DataFrame): Features for Stream B.
        """
        print("\n=== Generating Submission ===")

        results = []

        # --- Stream A Predictions ---
        if "A" in self.models and not X_test_a.empty:
            print(f"Predicting Stream A ({len(X_test_a)} samples)...")
            probs_a = self.models["A"].predict_proba(X_test_a)[:, 1]
            preds_a = (probs_a >= self.thresholds["A"]).astype(int)

            df_res_a = pd.DataFrame({"contact_id": ids_test_a, "contact": preds_a})
            results.append(df_res_a)
        elif not X_test_a.empty:
            print("Warning: Stream A data present but no model trained. Predicting 0.")
            results.append(pd.DataFrame({"contact_id": ids_test_a, "contact": 0}))

        # --- Stream B Predictions ---
        if "B" in self.models and not X_test_b.empty:
            print(f"Predicting Stream B ({len(X_test_b)} samples)...")
            probs_b = self.models["B"].predict_proba(X_test_b)[:, 1]
            preds_b = (probs_b >= self.thresholds["B"]).astype(int)

            df_res_b = pd.DataFrame({"contact_id": ids_test_b, "contact": preds_b})
            results.append(df_res_b)
        elif not X_test_b.empty:
            print("Warning: Stream B data present but no model trained. Predicting 0.")
            results.append(pd.DataFrame({"contact_id": ids_test_b, "contact": 0}))

        # --- Combine and Save ---
        if results:
            df_submission = pd.concat(results, axis=0)
        else:
            # Fallback for empty test set (unlikely)
            df_submission = pd.DataFrame(columns=["contact_id", "contact"])

        # Ensure unique contact_ids (though they should be unique by split)
        # and match the sample submission format if needed.
        # The competition usually requires all IDs from sample_submission to be present.
        # We assume the input X_test contains all necessary IDs.

        save_path = Config.SUBMISSION_PATH
        print(f"Saving submission to {save_path}...")
        df_submission.to_csv(save_path, index=False)
        print(f"Submission saved. Shape: {df_submission.shape}")

    def load_models(self):
        """
        Loads trained models from disk if they exist.
        Useful for inference-only runs.
        """
        for stream in ["A", "B"]:
            model_path = os.path.join(self.working_dir, f"model_stream_{stream}.json")
            if os.path.exists(model_path):
                clf = xgb.XGBClassifier()
                clf.load_model(model_path)
                self.models[stream] = clf
                # Note: Thresholds are not saved in the JSON model file.
                # In a real scenario, we'd save them in a separate config or pickle.
                # For this implementation, we assume training happens before prediction
                # or we use a default if loading solely from disk without context.
                if stream not in self.thresholds:
                    self.thresholds[stream] = 0.5  # Default fallback
