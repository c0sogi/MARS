import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from library.config import PathConfig, ModelConfig
from library.utils import calculate_log_loss


class LengthAdaptiveBlender:
    """
    Implements a Length-Adaptive Hybrid Ensembling strategy.

    This class addresses the correlation between model performance and text length by:
    1. Partitioning the validation set into bins based on character length (Short, Medium, Long).
    2. Optimizing blending weights (Statistical vs Neural models) independently for each bin
       to minimize Log Loss.
    3. Applying these bin-specific weights to the test set predictions.
    """

    def __init__(self, n_bins=3, seed=42):
        """
        Args:
            n_bins (int): Number of length bins to create (default: 3).
            seed (int): Random seed for reproducibility.
        """
        self.n_bins = n_bins
        self.seed = seed
        self.bin_thresholds = []
        self.bin_weights = {}  # Stores optimized weights for each bin index
        self.model_names = []

    def _get_text_lengths(self, texts):
        """
        Calculates character lengths for a collection of texts.

        Args:
            texts (list or pd.Series): Input texts.

        Returns:
            np.ndarray: Array of character lengths.
        """
        # Ensure texts are strings and handle potential non-string types gracefully
        return np.array([len(str(t)) for t in texts])

    def _optimize_weights_for_bin(self, preds_dict, y_true):
        """
        Optimizes blending weights for a specific bin using SLSQP to minimize Log Loss.

        Args:
            preds_dict (dict): Dictionary mapping model names to probability arrays (N, C).
            y_true (np.ndarray): True labels (N,).

        Returns:
            dict: Optimized weights {model_name: weight}.
        """
        model_names = list(preds_dict.keys())
        # Stack predictions: shape (n_models, n_samples, n_classes)
        predictions = np.array([preds_dict[m] for m in model_names])

        n_models = len(model_names)

        # Objective Function: Log Loss of weighted average
        def loss_func(weights):
            # Broadcast weights: (n_models, 1, 1)
            w = weights.reshape(-1, 1, 1)
            # Weighted sum: (n_samples, n_classes)
            weighted_pred = np.sum(w * predictions, axis=0)
            # Calculate metric
            return calculate_log_loss(y_true, weighted_pred)

        # Initial guess: Equal weights
        initial_weights = np.ones(n_models) / n_models

        # Constraints: Sum of weights must be 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Bounds: Weights must be between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Optimization
        result = minimize(
            loss_func,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=1e-6,
        )

        # Map results back to model names
        return {name: w for name, w in zip(model_names, result.x)}

    def fit(self, oof_preds_dict, y_true, val_texts):
        """
        Fits the blender: determines length thresholds and optimizes weights per bin.

        Args:
            oof_preds_dict (dict): Dictionary {model_name: np.ndarray (N_val, 3)}.
            y_true (np.ndarray): True label indices (N_val,).
            val_texts (pd.Series or list): Validation text content (N_val,).
        """
        self.model_names = list(oof_preds_dict.keys())
        lengths = self._get_text_lengths(val_texts)

        # 1. Determine Bin Thresholds (Quantiles)
        # For n_bins=3, we want 33.3% and 66.6% percentiles
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        # Extract inner boundaries
        self.bin_thresholds = np.percentile(lengths, quantiles[1:-1])

        print(
            f"Length-Adaptive Blending: Calculated Thresholds (Chars): {self.bin_thresholds}"
        )

        # 2. Assign samples to bins
        # np.digitize returns indices 0..n_bins-1 (if bins are defined by n_bins-1 thresholds)
        bin_indices = np.digitize(lengths, self.bin_thresholds)

        # 3. Optimize for each bin
        for bin_idx in range(self.n_bins):
            mask = bin_indices == bin_idx
            n_samples = np.sum(mask)

            if n_samples == 0:
                print(f"Warning: Bin {bin_idx} is empty. Defaulting to equal weights.")
                self.bin_weights[bin_idx] = {
                    m: 1.0 / len(self.model_names) for m in self.model_names
                }
                continue

            # Extract data for this bin
            bin_preds = {k: v[mask] for k, v in oof_preds_dict.items()}
            bin_y = (
                y_true[mask]
                if isinstance(y_true, np.ndarray)
                else np.array(y_true)[mask]
            )

            print(
                f"Optimizing Bin {bin_idx} (Size: {n_samples}, Range: "
                f"{'<' if bin_idx==0 else '>='} "
                f"{self.bin_thresholds[bin_idx-1] if bin_idx > 0 else ''} "
                f"{'to' if 0 < bin_idx < self.n_bins-1 else ''} "
                f"{'<' if bin_idx < self.n_bins-1 else ''} "
                f"{self.bin_thresholds[bin_idx] if bin_idx < self.n_bins-1 else ''})..."
            )

            weights = self._optimize_weights_for_bin(bin_preds, bin_y)
            self.bin_weights[bin_idx] = weights

            # Log weights
            w_str = ", ".join([f"{k}: {v:.4f}" for k, v in weights.items()])
            print(f"  -> Optimal Weights: {w_str}")

    def predict(self, test_preds_dict, test_texts):
        """
        Generates blended predictions for the test set using bin-specific weights.

        Args:
            test_preds_dict (dict): Dictionary {model_name: np.ndarray (N_test, 3)}.
            test_texts (pd.Series or list): Test text content (N_test,).

        Returns:
            np.ndarray: Final blended probabilities (N_test, 3).
        """
        lengths = self._get_text_lengths(test_texts)
        bin_indices = np.digitize(lengths, self.bin_thresholds)

        n_samples = len(lengths)
        # Assume all models have same shape
        n_classes = list(test_preds_dict.values())[0].shape[1]

        final_preds = np.zeros((n_samples, n_classes))

        # Apply weights per bin
        for bin_idx in range(self.n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) == 0:
                continue

            weights = self.bin_weights.get(bin_idx)

            # Fallback if weights missing (should not happen if fit called correctly)
            if weights is None:
                weights = {m: 1.0 / len(self.model_names) for m in self.model_names}

            # Compute weighted average for this subset
            weighted_chunk = np.zeros((np.sum(mask), n_classes))
            for model_name, w in weights.items():
                weighted_chunk += w * test_preds_dict[model_name][mask]

            final_preds[mask] = weighted_chunk

        return final_preds

    def generate_submission(
        self, ids, probabilities, output_path=PathConfig.SUBMISSION_FILE
    ):
        """
        Formats and saves the submission file.

        Args:
            ids (list or np.ndarray): Sample IDs.
            probabilities (np.ndarray): Predicted probabilities (N, 3).
            output_path (str): File path to save CSV.
        """
        # Ensure rows sum to 1 (normalization)
        row_sums = probabilities.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        probabilities = probabilities / row_sums[:, np.newaxis]

        # Create DataFrame
        df = pd.DataFrame(probabilities, columns=ModelConfig.LABELS)
        df.insert(0, "id", ids)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Submission successfully saved to {output_path}")
