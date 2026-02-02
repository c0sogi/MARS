import numpy as np
from sklearn.metrics import log_loss
from collections import Counter
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("ensemble_selection")


class GreedyEnsembleSelector:
    """
    Implements Greedy Forward Selection (Hill Climbing) to optimize ensemble weights.

    This strategy iteratively constructs an ensemble by adding the model that
    maximizes the performance improvement on the validation set. It allows for
    selection with replacement, which effectively learns integer weights for
    the experts.

    Reference: Caruana et al., "Ensemble Selection from Libraries of Models".
    """

    def __init__(self, n_iterations=100, tolerance=1e-6):
        """
        Initialize the selector.

        Args:
            n_iterations (int): Maximum number of iterations (maximum ensemble size).
                                Defaults to 100.
            tolerance (float): Minimum improvement in Log Loss required to continue
                               adding models. Defaults to 1e-6.
        """
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.selected_experts = []  # List of model names (allowing duplicates)
        self.best_score = float("inf")
        self.weights = {}

    def _normalize_and_clip(self, preds):
        """
        Applies the competition metric specific normalization and clipping.

        According to the metric definition:
        1. "The submitted probabilities... are rescaled prior to being scored
           (each row is divided by the row sum)".
        2. "predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)".

        Args:
            preds (np.ndarray): Raw probability matrix.

        Returns:
            np.ndarray: Normalized and clipped probabilities.
        """
        # Ensure float64 for precision
        preds = preds.astype(np.float64)

        # 1. Rescale rows to sum to 1
        row_sums = preds.sum(axis=1)
        # Avoid division by zero (handle rows with all zeros if any)
        row_sums[row_sums == 0] = 1.0
        preds_norm = preds / row_sums[:, np.newaxis]

        # 2. Clip values to avoid log(0)
        eps = 1e-15
        preds_clipped = np.clip(preds_norm, eps, 1 - eps)

        return preds_clipped

    def fit(self, predictions_dict, y_true):
        """
        Selects the optimal subset of experts using Greedy Forward Selection.

        Args:
            predictions_dict (dict): Dictionary where keys are model names and
                                     values are np.ndarray of shape (N_samples, N_classes)
                                     containing validation probabilities.
            y_true (np.ndarray): Ground truth labels of shape (N_samples,).

        Returns:
            self: The fitted instance.
        """
        logger.info(
            f"Starting Greedy Ensemble Selection with {len(predictions_dict)} candidates..."
        )

        # Validate inputs
        if not predictions_dict:
            raise ValueError("predictions_dict is empty.")

        model_names = list(predictions_dict.keys())
        n_samples = y_true.shape[0]
        # Get number of classes from the first prediction array
        n_classes = list(predictions_dict.values())[0].shape[1]

        # Initialize ensemble sum accumulator
        # We accumulate sums to avoid re-summing the entire history every iteration
        current_ensemble_sum = np.zeros((n_samples, n_classes), dtype=np.float64)

        self.selected_experts = []
        self.best_score = float("inf")

        # Iterative selection loop
        for i in range(self.n_iterations):
            iteration_best_score = float("inf")
            iteration_best_model = None

            # Try adding each candidate model to the current ensemble
            for name in model_names:
                candidate_preds = predictions_dict[name]

                # Calculate temporary ensemble average
                # New Avg = (Current Sum + Candidate Preds) / (Current Count + 1)
                temp_sum = current_ensemble_sum + candidate_preds
                temp_avg = temp_sum / (len(self.selected_experts) + 1)

                # Normalize and clip before scoring to match metric exactly
                final_preds = self._normalize_and_clip(temp_avg)

                # Calculate Log Loss
                # sklearn log_loss handles integer y_true and probability y_pred
                score = log_loss(y_true, final_preds)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_model = name

            # Check for improvement
            # For the first iteration, we accept the best single model regardless of "improvement"
            if not self.selected_experts:
                improvement = float("inf")
            else:
                improvement = self.best_score - iteration_best_score

            # Update ensemble if improvement meets tolerance
            if improvement > self.tolerance:
                self.selected_experts.append(iteration_best_model)
                self.best_score = iteration_best_score
                # Update the running sum
                current_ensemble_sum += predictions_dict[iteration_best_model]

                logger.info(
                    f"Iter {i+1}/{self.n_iterations}: Added {iteration_best_model}, "
                    f"Score: {self.best_score:.10f}, Improvement: {improvement:.10f}"
                )
            else:
                logger.info(
                    f"Iter {i+1}: No sufficient improvement ({improvement:.10f} <= {self.tolerance}). Stopping."
                )
                break

        # Calculate final weights based on selection frequency
        counts = Counter(self.selected_experts)
        total_selected = len(self.selected_experts)

        if total_selected == 0:
            logger.warning(
                "No experts were selected! This implies all models failed to converge or data is invalid."
            )
            self.weights = {}
        else:
            self.weights = {k: v / total_selected for k, v in counts.items()}

        logger.info("Ensemble Selection Complete.")
        logger.info(f"Final Weights: {self.weights}")

        return self

    def get_selected_experts(self):
        """
        Returns the dictionary of selected experts and their relative weights.

        Returns:
            dict: {model_name: weight}
        """
        return self.weights

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using the selected experts.

        Args:
            predictions_dict (dict): Dictionary of predictions (e.g., for test set).
                                     Keys must match those used in fit().

        Returns:
            np.ndarray: Aggregated, normalized, and clipped probabilities.
        """
        if not self.weights:
            raise RuntimeError(
                "EnsembleSelector is not fitted or no experts were selected."
            )

        # Determine shape from the first available prediction
        first_key = list(predictions_dict.keys())[0]
        sample_shape = predictions_dict[first_key].shape

        weighted_sum = np.zeros(sample_shape, dtype=np.float64)

        # Aggregate predictions
        for name, weight in self.weights.items():
            if name not in predictions_dict:
                logger.warning(
                    f"Selected expert '{name}' not found in predictions dictionary. Skipping."
                )
                continue

            # Ensure float64
            preds = predictions_dict[name].astype(np.float64)
            weighted_sum += preds * weight

        # Final normalization and clipping
        final_preds = self._normalize_and_clip(weighted_sum)

        return final_preds
