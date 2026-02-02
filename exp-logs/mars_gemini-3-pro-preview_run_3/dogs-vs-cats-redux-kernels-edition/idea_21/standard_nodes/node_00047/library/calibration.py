import numpy as np
import os
from sklearn.isotonic import IsotonicRegression
from library.utils import get_logger

# Initialize logger
logger = get_logger("calibration")


class Calibrator:
    """
    A class to handle Post-Hoc Calibration of model predictions.
    Utilizes Isotonic Regression to map predicted probabilities to
    better estimate the true probability distribution, minimizing Log Loss.
    """

    def __init__(self, method="isotonic"):
        """
        Initializes the Calibrator.

        Args:
            method (str): The calibration method to use. Currently supports 'isotonic'.
        """
        self.method = method
        self.model = None

    def fit(self, y_pred, y_true):
        """
        Fits the calibration model using Out-Of-Fold (OOF) predictions and ground truth labels.

        Args:
            y_pred (array-like): Raw predicted probabilities from the model (OOF).
            y_true (array-like): Ground truth binary labels (0 or 1).

        Returns:
            self: Returns the instance itself.
        """
        # Ensure inputs are 1D arrays
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()

        if self.method == "isotonic":
            # Isotonic Regression: Non-parametric approach to fit a non-decreasing function.
            # out_of_bounds="clip" ensures that if test probabilities fall outside
            # the range seen during training, they are clipped to the min/max seen.
            # y_min=0.0, y_max=1.0 constrains the output to valid probability range.
            self.model = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True
            )
            self.model.fit(y_pred, y_true)
            logger.info(
                f"Calibrator fitted using {self.method} on {len(y_pred)} samples."
            )
        else:
            logger.warning(
                f"Calibration method '{self.method}' not implemented. Calibrator will pass through predictions."
            )
            self.model = None

        return self

    def transform(self, y_pred):
        """
        Applies the fitted calibration model to new predictions (e.g., Test set).

        Args:
            y_pred (array-like): Raw predicted probabilities to calibrate.

        Returns:
            np.ndarray: Calibrated probabilities.
        """
        y_pred = np.array(y_pred).flatten()

        if self.model is None:
            # Fallback: Return original predictions if not fitted or method invalid
            return y_pred

        # Transform using the fitted Isotonic Regression model
        calibrated_preds = self.model.transform(y_pred)

        # Explicitly clip to ensure numerical stability within [0, 1]
        calibrated_preds = np.clip(calibrated_preds, 0.0, 1.0)

        return calibrated_preds


def apply_calibration(oof_preds, oof_targets, test_preds, method="isotonic"):
    """
    Functional interface to fit a calibrator on OOF data and transform test data.

    Args:
        oof_preds (array-like): Predictions on the validation set (Out-Of-Fold).
        oof_targets (array-like): Ground truth labels for the validation set.
        test_preds (array-like): Predictions on the test set.
        method (str): Calibration method ('isotonic').

    Returns:
        np.ndarray: Calibrated test predictions.
    """
    calibrator = Calibrator(method=method)
    calibrator.fit(oof_preds, oof_targets)
    calibrated_test_preds = calibrator.transform(test_preds)
    return calibrated_test_preds
