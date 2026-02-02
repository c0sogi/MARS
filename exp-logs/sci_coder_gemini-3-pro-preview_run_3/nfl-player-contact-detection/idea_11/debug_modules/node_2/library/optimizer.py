import numpy as np
from library.config import Config
from library.utils import compute_mcc


class ThresholdOptimizer:
    """
    Optimizer module for finding the best classification thresholds
    to maximize the Matthews Correlation Coefficient (MCC).

    This class performs a linear search over a range of probability thresholds
    defined in the Config, independently for Stream A and Stream B.
    """

    def __init__(self):
        """
        Initializes the optimizer with search space parameters from the global Config.
        """
        self.start = Config.THRESHOLD_SEARCH_START
        self.end = Config.THRESHOLD_SEARCH_END
        self.step = Config.THRESHOLD_SEARCH_STEP

    def optimize_single_stream(self, y_true, y_probs, stream_name="Stream"):
        """
        Optimizes the probability threshold for a single stream to maximize MCC.

        Args:
            y_true (array-like): Ground truth binary labels.
            y_probs (array-like): Predicted probabilities for the positive class.
            stream_name (str): Name of the stream for logging purposes.

        Returns:
            float: The optimal threshold value.
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        best_mcc = -1.0
        best_threshold = 0.5

        # Generate search range
        # np.arange excludes the endpoint, consistent with standard Python loops
        thresholds = np.arange(self.start, self.end, self.step)

        for t in thresholds:
            # Apply threshold to generate binary predictions
            preds = (y_probs >= t).astype(int)

            # Compute MCC
            score = compute_mcc(y_true, preds)

            # Update best score
            if score > best_mcc:
                best_mcc = score
                best_threshold = t

        print(f"--- {stream_name} Optimization Results ---")
        print(f"Optimal Threshold: {best_threshold}")
        print(f"Max MCC: {best_mcc}")

        return best_threshold

    def optimize_thresholds(self, y_true_a, y_probs_a, y_true_b, y_probs_b):
        """
        Performs threshold optimization for both Stream A (Interaction) and Stream B (Impact).

        Args:
            y_true_a (array-like): Ground truth labels for Stream A.
            y_probs_a (array-like): Predicted probabilities for Stream A.
            y_true_b (array-like): Ground truth labels for Stream B.
            y_probs_b (array-like): Predicted probabilities for Stream B.

        Returns:
            tuple: A tuple containing (best_threshold_a, best_threshold_b).
        """
        print("Starting Threshold Optimization...")

        # Optimize Stream A
        thresh_a = self.optimize_single_stream(y_true_a, y_probs_a, "Stream A")

        # Optimize Stream B
        thresh_b = self.optimize_single_stream(y_true_b, y_probs_b, "Stream B")

        return thresh_a, thresh_b
