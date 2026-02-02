import numpy as np
from library.utils import compute_mcc


class ThresholdOptimizer:
    """
    Handles the post-training optimization of decision boundaries for the Dual-Stream architecture.
    Performs independent linear searches for Stream A and Stream B thresholds to maximize
    performance, then evaluates the combined Matthews Correlation Coefficient (MCC).
    """

    def __init__(self, steps=100):
        """
        Initialize the optimizer.

        Args:
            steps (int): Number of steps in the linear search for thresholds (default: 100).
                         This defines the resolution of the grid between 0.0 and 1.0.
        """
        self.steps = steps

    def _optimize_single_stream(self, y_true, y_probs, stream_name):
        """
        Finds the optimal threshold for a single stream by maximizing MCC.

        Args:
            y_true (np.ndarray): Ground truth labels (0 or 1).
            y_probs (np.ndarray): Predicted probabilities.
            stream_name (str): Name of the stream for logging (e.g., "A" or "B").

        Returns:
            tuple: (best_threshold, best_mcc)
        """
        best_threshold = 0.5
        best_mcc = -1.0

        # Define search space: 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, self.steps)

        for thresh in thresholds:
            # Convert probabilities to binary predictions
            preds = (y_probs >= thresh).astype(int)

            # Compute metric
            score = compute_mcc(y_true, preds)

            if score > best_mcc:
                best_mcc = score
                best_threshold = thresh

        print(f"Stream {stream_name} Optimization Results:")
        print(f"  Best Threshold: {best_threshold}")
        print(f"  Best MCC: {best_mcc}")

        return best_threshold, best_mcc

    def optimize_thresholds(self, y_true_a, y_probs_a, y_true_b, y_probs_b):
        """
        Performs a linear search on the validation set probabilities for both Stream A and
        Stream B independently to find the cutoffs that maximize their respective MCCs,
        and then calculates the combined MCC.

        Args:
            y_true_a (np.ndarray): Ground truth labels for Stream A.
            y_probs_a (np.ndarray): Predicted probabilities for Stream A.
            y_true_b (np.ndarray): Ground truth labels for Stream B.
            y_probs_b (np.ndarray): Predicted probabilities for Stream B.

        Returns:
            tuple: (thresh_a, thresh_b, combined_mcc)
        """
        print("Starting Threshold Optimization...")

        # 1. Optimize Stream A (Player-Player)
        thresh_a, mcc_a = self._optimize_single_stream(y_true_a, y_probs_a, "A")

        # 2. Optimize Stream B (Player-Ground)
        thresh_b, mcc_b = self._optimize_single_stream(y_true_b, y_probs_b, "B")

        # 3. Calculate Combined MCC
        # Concatenate ground truths
        y_true_all = np.concatenate([y_true_a, y_true_b])

        # Generate predictions using respective optimal thresholds
        preds_a = (y_probs_a >= thresh_a).astype(int)
        preds_b = (y_probs_b >= thresh_b).astype(int)
        y_preds_all = np.concatenate([preds_a, preds_b])

        # Compute global metric
        combined_mcc = compute_mcc(y_true_all, y_preds_all)

        print("-" * 30)
        print("Combined Optimization Results:")
        print(f"  Threshold A: {thresh_a}")
        print(f"  Threshold B: {thresh_b}")
        print(f"  Combined MCC: {combined_mcc}")
        print("-" * 30)

        return thresh_a, thresh_b, combined_mcc
