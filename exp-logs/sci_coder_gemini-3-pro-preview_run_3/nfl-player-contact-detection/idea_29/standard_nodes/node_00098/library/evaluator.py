import numpy as np
from sklearn.metrics import matthews_corrcoef
from library.config import Config


class Evaluator:
    """
    Evaluator module for calculating metrics and optimizing thresholds
    for the Hybrid-Context Dual-Stream GBDT.
    """

    @staticmethod
    def optimize_threshold(
        y_true: np.ndarray, y_probs: np.ndarray, stream_name: str = "Unknown"
    ) -> float:
        """
        Performs a linear search on the validation set predictions to find the
        probability threshold that maximizes the Matthews Correlation Coefficient (MCC).

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_probs (np.ndarray): Predicted probabilities for the positive class.
            stream_name (str): Identifier for the stream (e.g., 'A' or 'B').

        Returns:
            float: The probability threshold that maximizes MCC.
        """
        print(f"\n--- Optimizing Threshold for Stream {stream_name} ---")

        # Retrieve search space from Config
        start, stop, step = Config.THRESHOLD_SEARCH
        # np.arange creates values within [start, stop) with step size
        thresholds = np.arange(start, stop, step)

        best_threshold = 0.5
        best_mcc = -1.0

        # Iterate through thresholds to find the optimal one
        for thresh in thresholds:
            # Convert probabilities to binary predictions based on current threshold
            preds = (y_probs >= thresh).astype(int)

            # Calculate MCC
            mcc = matthews_corrcoef(y_true, preds)

            # Update best metric if current MCC is higher
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh

        # Print results with full precision (no formatting specifiers)
        print(f"Stream {stream_name} Best Threshold: {best_threshold}")
        print(f"Stream {stream_name} Best Validation MCC: {best_mcc}")

        return best_threshold

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float,
        stream_name: str = "Unknown",
    ) -> float:
        """
        Calculates MCC for a specific threshold.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_probs (np.ndarray): Predicted probabilities.
            threshold (float): The threshold to apply.
            stream_name (str): Identifier for the stream.

        Returns:
            float: The MCC score.
        """
        preds = (y_probs >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        print(f"Stream {stream_name} MCC at threshold {threshold}: {mcc}")
        return mcc
