import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from library.config import Config
from library.feature_extractor import FeatureExtractor


class StreamClassifier:
    """
    Manages the training of stream-specific classifiers, ensemble optimization,
    and submission generation.
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.classes = self._load_classes()
        self._set_seed()

    def _set_seed(self):
        np.random.seed(Config.SEED)

    def _load_classes(self):
        """
        Replicates the class loading logic from DogDataset to ensure
        column headers match prediction indices.
        """
        if not os.path.exists(Config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(f"Metadata not found: {Config.TRAIN_METADATA_PATH}")

        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        classes = sorted(train_df["breed"].unique().tolist())
        return classes

    def train_stream(self, stream_name, debug_sample_size=None):
        """
        Trains a LogisticRegressionCV classifier for a specific stream.

        Args:
            stream_name (str): 'stream_a' or 'stream_b'.
            debug_sample_size (int, optional): Limit data size for debugging.

        Returns:
            model: The trained sklearn model.
            val_probs: Validation probabilities (N_val, N_classes).
            val_labels: Ground truth validation labels.
        """
        print(f"\n--- Training Classifier for {stream_name} ---")

        # 1. Get Data
        # FeatureExtractor handles caching and concatenation of views
        X_train, y_train, _ = self.feature_extractor.extract_features(
            stream_name, "train", debug_sample_size=debug_sample_size
        )
        X_val, y_val, _ = self.feature_extractor.extract_features(
            stream_name, "val", debug_sample_size=debug_sample_size
        )

        print(f"  Train shape: {X_train.shape}, Labels: {y_train.shape}")
        print(f"  Val shape:   {X_val.shape}, Labels: {y_val.shape}")

        # 2. Define Model
        # LogisticRegressionCV automatically tunes 'C'
        clf = LogisticRegressionCV(
            Cs=Config.CS_COUNT,
            cv=Config.CV_FOLDS,
            penalty="l2",
            scoring="neg_log_loss",
            solver="lbfgs",
            max_iter=Config.MAX_ITER,
            multi_class="multinomial",
            n_jobs=Config.NUM_WORKERS,
            random_state=Config.SEED,
            verbose=0,
        )

        # 3. Train
        print(f"  Fitting LogisticRegressionCV...")
        clf.fit(X_train, y_train)

        # 4. Evaluate
        print(
            f"  Best C: {clf.C_[0]}"
        )  # C_ is array of shape (n_classes,) or (1,) depending on binary/multi

        val_probs = clf.predict_proba(X_val)
        loss = log_loss(y_val, val_probs)
        print(f"  Validation Log Loss ({stream_name}): {loss}")

        # 5. Save Model
        save_path = (
            Config.MODEL_A_HEAD_PATH
            if stream_name == "stream_a"
            else Config.MODEL_B_HEAD_PATH
        )
        print(f"  Saving model to {save_path}")
        joblib.dump(clf, save_path)

        return clf, val_probs, y_val

    def optimize_ensemble(self, probs_a, probs_b, y_true):
        """
        Finds the optimal weight w for the ensemble: P = w * P_a + (1-w) * P_b
        """
        print("\n--- Optimizing Ensemble Weights ---")

        def objective(w):
            w = w[0]
            # Clip w to [0, 1] conceptually, though bounds handle it
            p_ensemble = w * probs_a + (1 - w) * probs_b
            # Clip probabilities to avoid log(0)
            p_ensemble = np.clip(p_ensemble, 1e-15, 1 - 1e-15)
            return log_loss(y_true, p_ensemble)

        # Initial guess: 0.5
        initial_w = [0.5]
        bounds = [(0.0, 1.0)]

        result = minimize(objective, initial_w, bounds=bounds, method="L-BFGS-B")

        best_w = result.x[0]
        best_loss = result.fun

        print(f"  Optimal Weight for Stream A: {best_w}")
        print(f"  Optimal Weight for Stream B: {1 - best_w}")
        print(f"  Combined Validation Log Loss: {best_loss}")

        # Save weights
        weights_data = {
            "w_a": float(best_w),
            "w_b": float(1 - best_w),
            "val_loss": float(best_loss),
        }
        with open(Config.ENSEMBLE_WEIGHTS_PATH, "w") as f:
            json.dump(weights_data, f, indent=4)

        return weights_data

    def generate_submission(self, model_a, model_b, weights, debug_sample_size=None):
        """
        Generates predictions for the test set and creates the submission file.
        """
        print("\n--- Generating Submission ---")

        # 1. Get Test Data
        X_test_a, _, ids_test_a = self.feature_extractor.extract_features(
            "stream_a", "test", debug_sample_size=debug_sample_size
        )
        X_test_b, _, ids_test_b = self.feature_extractor.extract_features(
            "stream_b", "test", debug_sample_size=debug_sample_size
        )

        # Verify ID alignment
        if not np.array_equal(ids_test_a, ids_test_b):
            raise ValueError("Mismatch in test IDs between Stream A and Stream B")

        # 2. Predict
        print("  Predicting Stream A...")
        probs_a = model_a.predict_proba(X_test_a)

        print("  Predicting Stream B...")
        probs_b = model_b.predict_proba(X_test_b)

        # 3. Ensemble
        w_a = weights["w_a"]
        w_b = weights["w_b"]
        final_probs = w_a * probs_a + w_b * probs_b

        # 4. Create DataFrame
        # Columns must be the breed names in sorted order
        df = pd.DataFrame(final_probs, columns=self.classes)
        df.insert(0, "id", ids_test_a)

        # 5. Save
        print(f"  Saving submission to {Config.SUBMISSION_PATH}")
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("  Submission generated successfully.")

    def run(self, debug_sample_size=None):
        """
        Orchestrates the full pipeline.
        """
        # Train Stream A
        model_a, val_probs_a, val_labels_a = self.train_stream(
            "stream_a", debug_sample_size=debug_sample_size
        )

        # Train Stream B
        model_b, val_probs_b, val_labels_b = self.train_stream(
            "stream_b", debug_sample_size=debug_sample_size
        )

        # Verify label alignment
        if not np.array_equal(val_labels_a, val_labels_b):
            raise ValueError("Mismatch in validation labels between streams.")

        # Optimize Ensemble
        weights = self.optimize_ensemble(val_probs_a, val_probs_b, val_labels_a)

        # Generate Submission
        self.generate_submission(
            model_a, model_b, weights, debug_sample_size=debug_sample_size
        )
