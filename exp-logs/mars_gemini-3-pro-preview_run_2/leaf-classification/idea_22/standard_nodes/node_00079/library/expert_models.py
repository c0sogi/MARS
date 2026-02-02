import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from library.config import Config


def get_lda_expert(solver=None, shrinkage=None):
    """
    Constructs the Linear Discriminant Analysis expert.

    This model serves as the backbone for the Anchor, Orthogonal, and Synergistic experts.
    It uses Ledoit-Wolf shrinkage ('auto' with 'lsqr' solver) to estimate covariance
    matrices robustly in high-dimensional settings.

    Args:
        solver (str, optional): Solver to use. Defaults to Config.LDA_PARAMS['solver'].
        shrinkage (str/float, optional): Shrinkage parameter. Defaults to Config.LDA_PARAMS['shrinkage'].

    Returns:
        LinearDiscriminantAnalysis: The configured LDA estimator.
    """
    # Load defaults from Config
    params = Config.LDA_PARAMS.copy()

    # Override if arguments provided
    if solver is not None:
        params["solver"] = solver
    if shrinkage is not None:
        params["shrinkage"] = shrinkage

    return LinearDiscriminantAnalysis(**params)


def get_lr_expert(C=None, calibration_method="sigmoid", cv=5):
    """
    Constructs the Discriminative Backup expert: Calibrated Logistic Regression.

    This model provides a discriminative safety net. It is wrapped in CalibratedClassifierCV
    to ensuring the output probabilities are comparable to the generative LDA models.

    Args:
        C (float, optional): Inverse of regularization strength.
                             If provided, overrides Config.LOGREG_PARAMS['C'].
                             This allows passing the optimal C found during selection.
        calibration_method (str, optional): Method for calibration ('sigmoid' or 'isotonic').
                                            Defaults to 'sigmoid'.
        cv (int, optional): Cross-validation generator or count for calibration.
                            Defaults to 5.

    Returns:
        CalibratedClassifierCV: The configured calibrated logistic regression estimator.
    """
    # Load defaults from Config
    params = Config.LOGREG_PARAMS.copy()

    # Set Random State for reproducibility
    params["random_state"] = Config.RANDOM_SEED

    # Override C if provided (crucial for Phase 2 retraining)
    if C is not None:
        params["C"] = C

    # Create the base estimator
    # Note: In scikit-learn 1.2+, the parameter is 'estimator' for CalibratedClassifierCV,
    # but the inner model is just instantiated here.
    base_lr = LogisticRegression(**params)

    # Wrap in CalibratedClassifierCV
    # We use the 'estimator' parameter which is standard in recent sklearn versions.
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_lr, method=calibration_method, cv=cv, n_jobs=Config.N_JOBS
    )

    return calibrated_clf
