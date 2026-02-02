import numpy as np
from sklearn.metrics import matthews_corrcoef
from library.utils import set_seed
from library.config import Config


class Evaluator:
    """
    Handles metric calculation and threshold optimization for the Dual-Stream GBDT solution.
    Provides functionality to compute MCC and optimize probability thresholds
    independently for Player-Player (Stream A) and Player-Ground (Stream B) interactions.
    """

    def __init__(self):
        # Set seed for any potential numpy operations, though evaluation is largely deterministic
        set_seed(Config.SEED)

    @staticmethod
    def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Matthews Correlation Coefficient between true and predicted labels.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_pred (np.ndarray): Predicted binary labels.

        Returns:
            float: The MCC score.
        """
        return matthews_corrcoef(y_true, y_pred)

    def _optimize_single_stream(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Performs a linear search to find the probability threshold that maximizes MCC
        for a single data stream.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_proba (np.ndarray): Predicted probabilities.

        Returns:
            tuple: (best_threshold, best_mcc)
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Linear search from 0.01 to 0.99 with 99 steps
        thresholds = np.linspace(0.01, 0.99, 99)

        for thresh in thresholds:
            # Convert probabilities to binary predictions based on current threshold
            y_pred = (y_proba >= thresh).astype(int)

            # Calculate metric
            mcc = self.compute_mcc(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        return best_thresh, best_mcc

    def optimize_thresholds(
        self,
        y_true_a: np.ndarray,
        y_proba_a: np.ndarray,
        y_true_b: np.ndarray,
        y_proba_b: np.ndarray,
    ):
        """
        Optimizes thresholds independently for Stream A and Stream B using the validation set.
        Prints the best threshold and corresponding MCC for each stream with full precision.

        Args:
            y_true_a (np.ndarray): Ground truth labels for Stream A.
            y_proba_a (np.ndarray): Predicted probabilities for Stream A.
            y_true_b (np.ndarray): Ground truth labels for Stream B.
            y_proba_b (np.ndarray): Predicted probabilities for Stream B.

        Returns:
            dict: Dictionary containing best thresholds and scores for both streams.
                  Format: {'stream_a': {'threshold': float, 'mcc': float}, ...}
        """
        print("Optimizing thresholds for Stream A (Player-Player)...")
        thresh_a, mcc_a = self._optimize_single_stream(y_true_a, y_proba_a)
        print(f"Stream A - Best Threshold: {thresh_a} | Max MCC: {mcc_a}")

        print("Optimizing thresholds for Stream B (Player-Ground)...")
        thresh_b, mcc_b = self._optimize_single_stream(y_true_b, y_proba_b)
        print(f"Stream B - Best Threshold: {thresh_b} | Max MCC: {mcc_b}")

        return {
            "stream_a": {"threshold": thresh_a, "mcc": mcc_a},
            "stream_b": {"threshold": thresh_b, "mcc": mcc_b},
        }
