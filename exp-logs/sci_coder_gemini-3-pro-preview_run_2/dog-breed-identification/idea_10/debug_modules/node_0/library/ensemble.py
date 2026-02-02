import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from library.config import Config
from library.feature_extractor import FeatureExtractor


class EnsembleOptimizer:
    """
    Manages the ensemble optimization and submission generation for the
    Dual-Stream Heterogeneous Multi-View architecture.
    """

    def __init__(self):
        """
        Initialize the optimizer with a feature extractor and load class definitions.
        """
        self.feature_extractor = FeatureExtractor()
        self.classes = self._load_classes()

    def _load_classes(self):
        """
        Loads the unique classes from the training metadata to ensure consistency
        in the submission file headers.
        """
        if not os.path.exists(Config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Metadata file not found: {Config.TRAIN_METADATA_PATH}"
            )

        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        classes = sorted(train_df["breed"].unique().tolist())
        return classes

    def _load_model(self, stream_name):
        """
        Loads the trained classifier head for a specific stream.

        Args:
            stream_name (str): 'stream_a' or 'stream_b'.
        """
        path = (
            Config.MODEL_A_HEAD_PATH
            if stream_name == "stream_a"
            else Config.MODEL_B_HEAD_PATH
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file for {stream_name} not found at {path}")

        return joblib.load(path)

    def _get_predictions(self, stream_name, split_name, model, debug_sample_size=None):
        """
        Generates probability predictions for a specific stream and split.
        Uses FeatureExtractor to get embeddings (cached) and the model to predict.

        Args:
            stream_name (str): 'stream_a' or 'stream_b'.
            split_name (str): 'val' or 'test'.
            model: The trained scikit-learn model.
            debug_sample_size (int, optional): Limit data size for debugging.

        Returns:
            tuple: (probabilities, labels, ids)
        """
        # Extract features (FeatureExtractor handles caching and multi-view concatenation)
        embeddings, labels, ids = self.feature_extractor.extract_features(
            stream_name, split_name, debug_sample_size=debug_sample_size
        )

        # Predict probabilities
        probs = model.predict_proba(embeddings)

        return probs, labels, ids

    def optimize_weights(self, debug_sample_size=None):
        """
        Optimizes the ensemble weights using the validation set.
        Minimizes Log Loss using L-BFGS-B.
        Saves the optimal weights to Config.ENSEMBLE_WEIGHTS_PATH.

        Args:
            debug_sample_size (int, optional): Limit data size for debugging.

        Returns:
            dict: The optimized weights and validation loss.
        """
        print("--- Starting Ensemble Weight Optimization ---")

        # Load Models
        model_a = self._load_model("stream_a")
        model_b = self._load_model("stream_b")

        # Get Validation Predictions
        print("Generating validation predictions for Stream A...")
        probs_a, y_val_a, _ = self._get_predictions(
            "stream_a", "val", model_a, debug_sample_size
        )

        print("Generating validation predictions for Stream B...")
        probs_b, y_val_b, _ = self._get_predictions(
            "stream_b", "val", model_b, debug_sample_size
        )

        # Verify Alignment
        if not np.array_equal(y_val_a, y_val_b):
            raise ValueError(
                "Validation labels mismatch between Stream A and Stream B."
            )

        y_true = y_val_a

        # Optimization Objective
        def objective(w):
            w = w[0]
            # Weighted average of probabilities
            p_ensemble = w * probs_a + (1 - w) * probs_b
            # Clip probabilities to avoid log(0)
            p_ensemble = np.clip(p_ensemble, 1e-15, 1 - 1e-15)
            return log_loss(y_true, p_ensemble)

        # Initial guess: 0.5 (equal weight)
        initial_w = [0.5]
        # Bounds for w_a: [0, 1]
        bounds = [(0.0, 1.0)]

        print("Optimizing weights using L-BFGS-B...")
        result = minimize(objective, initial_w, bounds=bounds, method="L-BFGS-B")

        best_w_a = result.x[0]
        best_w_b = 1.0 - best_w_a
        best_loss = result.fun

        print(f"Optimization Results:")
        print(f"  Optimal Weight Stream A: {best_w_a}")
        print(f"  Optimal Weight Stream B: {best_w_b}")
        print(f"  Best Validation Log Loss: {best_loss}")

        # Save Weights
        weights_data = {
            "w_a": float(best_w_a),
            "w_b": float(best_w_b),
            "val_loss": float(best_loss),
        }

        with open(Config.ENSEMBLE_WEIGHTS_PATH, "w") as f:
            json.dump(weights_data, f, indent=4)

        print(f"Weights saved to {Config.ENSEMBLE_WEIGHTS_PATH}")
        return weights_data

    def generate_submission(self, weights=None, debug_sample_size=None):
        """
        Generates the final submission file using the optimized weights.

        Args:
            weights (dict, optional): Dictionary containing 'w_a' and 'w_b'.
                                      If None, loads from disk.
            debug_sample_size (int, optional): Limit data size for debugging.
        """
        print("\n--- Generating Final Submission ---")

        # Load Weights if not provided
        if weights is None:
            if os.path.exists(Config.ENSEMBLE_WEIGHTS_PATH):
                with open(Config.ENSEMBLE_WEIGHTS_PATH, "r") as f:
                    weights = json.load(f)
            else:
                raise FileNotFoundError(
                    "Weights not provided and file not found. Run optimize_weights first."
                )

        w_a = weights["w_a"]
        w_b = weights["w_b"]
        print(f"Using weights: Stream A = {w_a}, Stream B = {w_b}")

        # Load Models
        model_a = self._load_model("stream_a")
        model_b = self._load_model("stream_b")

        # Get Test Predictions
        print("Generating test predictions for Stream A...")
        probs_a, _, ids_a = self._get_predictions(
            "stream_a", "test", model_a, debug_sample_size
        )

        print("Generating test predictions for Stream B...")
        probs_b, _, ids_b = self._get_predictions(
            "stream_b", "test", model_b, debug_sample_size
        )

        # Verify Alignment
        if not np.array_equal(ids_a, ids_b):
            raise ValueError("Test IDs mismatch between Stream A and Stream B.")

        # Apply Ensemble Weights
        final_probs = w_a * probs_a + w_b * probs_b

        # Create Submission DataFrame
        df = pd.DataFrame(final_probs, columns=self.classes)
        df.insert(0, "id", ids_a)

        # Save to CSV
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")
