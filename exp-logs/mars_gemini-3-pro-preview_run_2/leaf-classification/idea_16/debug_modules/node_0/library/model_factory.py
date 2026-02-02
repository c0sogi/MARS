import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import library.config as conf


def get_expert_pool():
    """
    Constructs and returns the dictionary of 9 expert models (estimators)
    as defined in the Dynamic Multi-View Ensemble strategy.

    Returns:
        dict: A dictionary where keys are model identifiers (e.g., 'Global_LR', 'Texture_LDA')
              and values are uninitialized sklearn estimator objects.
    """
    pool = {}

    # =========================================================================
    # 1. Discriminative Linear Experts (Logistic Regression)
    # =========================================================================
    # One expert per view.
    # We use LogisticRegressionCV to automatically find the best C (regularization strength)
    # for each specific view, effectively decoupling regularization.
    for view_name in conf.VIEWS.keys():
        model_key = f"{view_name}_LR"
        pool[model_key] = LogisticRegressionCV(
            Cs=conf.LR_CS_GRID,
            cv=conf.LR_CV_FOLDS,
            solver=conf.LR_SOLVER,
            max_iter=conf.LR_MAX_ITER,
            scoring="neg_log_loss",
            random_state=conf.RANDOM_SEED,
            n_jobs=-1,
        )

    # =========================================================================
    # 2. Generative Linear Experts (Linear Discriminant Analysis)
    # =========================================================================
    # One expert per view.
    # We use Ledoit-Wolf shrinkage (shrinkage='auto' + solver='lsqr') to handle
    # potential high-dimensionality relative to class count and improve density estimation.
    for view_name in conf.VIEWS.keys():
        model_key = f"{view_name}_LDA"
        pool[model_key] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    # =========================================================================
    # 3. Discriminative Non-Linear Expert (Random Forest)
    # =========================================================================
    # Global view only.
    # Wrapped in CalibratedClassifierCV to fix the poor probability calibration
    # often seen in raw Random Forests.
    rf_base = RandomForestClassifier(
        n_estimators=conf.RF_N_ESTIMATORS, random_state=conf.RANDOM_SEED, n_jobs=-1
    )

    # Note: CalibratedClassifierCV with cv=k will fit k models.
    pool["Global_RF"] = CalibratedClassifierCV(
        estimator=rf_base,
        method=conf.CALIBRATION_METHOD,
        cv=conf.CALIBRATION_CV,
        n_jobs=-1,
    )

    return pool
