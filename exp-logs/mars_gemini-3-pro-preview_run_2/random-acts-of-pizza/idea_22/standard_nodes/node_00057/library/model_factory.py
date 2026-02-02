import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from library.config import SEED, BAGGING_CONFIG


def build_ensemble_pipeline(
    metadata_start_idx, lr_params=None, svm_params=None, ridge_params=None
):
    """
    Constructs the Heterogeneous Linear Ensemble pipeline.

    Args:
        metadata_start_idx (int): The column index where metadata features begin.
                                  Columns before this are assumed to be embeddings.
        lr_params (dict, optional): Hyperparameters for LogisticRegression.
        svm_params (dict, optional): Hyperparameters for SGDClassifier (SVM).
        ridge_params (dict, optional): Hyperparameters for RidgeClassifier.

    Returns:
        sklearn.pipeline.Pipeline: The complete modeling pipeline.
    """

    # ---------------------------------------------------------
    # 1. Default Hyperparameters (High-Regularization Regime)
    # ---------------------------------------------------------
    if lr_params is None:
        lr_params = {
            "solver": "liblinear",
            "penalty": "l2",
            "C": 0.1,
            "class_weight": "balanced",
            "random_state": SEED,
        }
    else:
        # Ensure random_state is set if not provided
        if "random_state" not in lr_params:
            lr_params["random_state"] = SEED

    if svm_params is None:
        svm_params = {
            "loss": "hinge",
            "penalty": "l2",
            "alpha": 0.01,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": SEED,
        }
    else:
        if "random_state" not in svm_params:
            svm_params["random_state"] = SEED

    if ridge_params is None:
        ridge_params = {"alpha": 10.0, "class_weight": "balanced", "random_state": SEED}
    else:
        if "random_state" not in ridge_params:
            ridge_params["random_state"] = SEED

    # ---------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------
    # Apply QuantileTransformer only to metadata columns (starting from metadata_start_idx).
    # Pass embeddings (columns 0 to metadata_start_idx) through unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "meta",
                QuantileTransformer(output_distribution="normal", random_state=SEED),
                slice(metadata_start_idx, None),
            ),
            ("embed", "passthrough", slice(0, metadata_start_idx)),
        ]
    )

    # ---------------------------------------------------------
    # 3. Base Learners Construction
    # ---------------------------------------------------------

    # Learner A: Bagged Logistic Regression (Log-Likelihood)
    # LR supports predict_proba natively.
    lr_base = LogisticRegression(**lr_params)
    lr_bagged = BaggingClassifier(estimator=lr_base, **BAGGING_CONFIG)

    # Learner B: Bagged Linear SVM (Hinge Loss)
    # SGDClassifier(loss='hinge') does not support predict_proba.
    # We wrap it in CalibratedClassifierCV to obtain probabilistic outputs via Platt scaling/Isotonic regression.
    svm_base = SGDClassifier(**svm_params)
    svm_calibrated = CalibratedClassifierCV(estimator=svm_base, cv=3, method="sigmoid")
    svm_bagged = BaggingClassifier(estimator=svm_calibrated, **BAGGING_CONFIG)

    # Learner C: Bagged Ridge Classifier (Squared Error)
    # RidgeClassifier outputs decision scores, not probabilities.
    # We wrap it in CalibratedClassifierCV.
    ridge_base = RidgeClassifier(**ridge_params)
    ridge_calibrated = CalibratedClassifierCV(
        estimator=ridge_base, cv=3, method="sigmoid"
    )
    ridge_bagged = BaggingClassifier(estimator=ridge_calibrated, **BAGGING_CONFIG)

    # ---------------------------------------------------------
    # 4. Ensemble
    # ---------------------------------------------------------
    # Soft Voting averages the probabilities from the three diverse linear approaches.
    ensemble = VotingClassifier(
        estimators=[("lr", lr_bagged), ("svm", svm_bagged), ("ridge", ridge_bagged)],
        voting="soft",
        n_jobs=1,  # Prevent nested parallelism issues; parallelism handled by outer loops or bagging
    )

    # ---------------------------------------------------------
    # 5. Final Pipeline
    # ---------------------------------------------------------
    pipeline = Pipeline([("preprocessor", preprocessor), ("ensemble", ensemble)])

    return pipeline
