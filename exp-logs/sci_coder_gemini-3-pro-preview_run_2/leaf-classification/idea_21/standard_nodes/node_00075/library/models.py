from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from library.config import LDA_PARAMS, LOGREG_PARAMS, CALIBRATION_PARAMS


class ExpertFactory:
    """
    Factory class to create un-fitted sklearn estimator objects for the
    Dynamic Ensemble Selection pipeline.
    """

    @staticmethod
    def create_lda_expert():
        """
        Creates a Linear Discriminant Analysis model with Ledoit-Wolf shrinkage.
        Used for Expert A (Original), Expert B (Morphological), and Expert C (Combined).

        Returns:
            LinearDiscriminantAnalysis: The configured LDA estimator.
        """
        # LDA with 'lsqr' solver and 'auto' shrinkage (Ledoit-Wolf)
        # This configuration is robust for high-dimensional data (small N, large P)
        return LinearDiscriminantAnalysis(**LDA_PARAMS)

    @staticmethod
    def create_calibrated_lr_expert():
        """
        Creates a Calibrated Logistic Regression model.
        Used for Expert D (Discriminative Backup).

        The base estimator is Logistic Regression (L2, L-BFGS).
        It is wrapped in CalibratedClassifierCV (Isotonic) to correct potential
        probability miscalibration.

        Returns:
            CalibratedClassifierCV: The configured calibrated estimator.
        """
        # Initialize the base Logistic Regression model with project hyperparameters
        # LOGREG_PARAMS includes random_state for reproducibility
        base_estimator = LogisticRegression(**LOGREG_PARAMS)

        # Wrap in CalibratedClassifierCV
        # CALIBRATION_PARAMS defines the method (isotonic) and CV folds
        calibrated_model = CalibratedClassifierCV(
            estimator=base_estimator, **CALIBRATION_PARAMS
        )

        return calibrated_model
