import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

from library.config import Config
from library.utils import calculate_metric


def create_hybrid_ensemble():
    """
    Constructs the Hybrid Kernel-Linear Ensemble consisting of:
    1. Logistic Regression (Discriminative Linear)
    2. LDA (Generative Linear)
    3. Calibrated SVM (Discriminative Non-Linear)

    Returns:
        VotingClassifier: The configured ensemble model.
    """
    # 1. Logistic Regression (Discriminative Linear Component)
    # Uses LogisticRegressionCV for efficient internal cross-validation of the C parameter.
    print("Initializing Logistic Regression component...")
    lr_clf = LogisticRegressionCV(**Config.LR_PARAMS)

    # 2. Linear Discriminant Analysis (Generative Linear Component)
    # Uses automatic covariance shrinkage (Ledoit-Wolf) to handle high dimensionality.
    print("Initializing LDA component...")
    lda_clf = LinearDiscriminantAnalysis(**Config.LDA_PARAMS)

    # 3. Calibrated SVM (Discriminative Non-Linear Component)
    # RBF Kernel SVM, tuned via GridSearchCV, and then probability-calibrated.
    print("Initializing SVM component...")

    # Base estimator
    base_svc = SVC(**Config.SVM_BASE_PARAMS)

    # Internal Hyperparameter Tuning
    # We wrap the base SVC in GridSearchCV to find optimal C and Gamma.
    # Note: We use default scoring (accuracy) for the inner loop as probability=False
    # on the base estimator, and we want to maximize the margin/decision boundary quality.
    svc_grid = GridSearchCV(
        estimator=base_svc, param_grid=Config.SVM_PARAM_GRID, cv=5, n_jobs=-1
    )

    # Calibration Wrapper
    # Transforms the uncalibrated decision function of the optimized SVM into probabilities.
    # This is crucial for the Log Loss metric.
    calibrated_svc = CalibratedClassifierCV(
        estimator=svc_grid, **Config.SVM_CALIBRATION_PARAMS
    )

    # 4. Ensemble Construction
    print("Constructing VotingClassifier...")
    estimators = [("lr", lr_clf), ("lda", lda_clf), ("svm", calibrated_svc)]

    # Extract weights ensuring correct order matching estimators list
    weights = [
        Config.ENSEMBLE_WEIGHTS["lr"],
        Config.ENSEMBLE_WEIGHTS["lda"],
        Config.ENSEMBLE_WEIGHTS["svm"],
    ]

    ensemble = VotingClassifier(
        estimators=estimators, voting="soft", weights=weights, n_jobs=-1
    )

    return ensemble


def train_and_evaluate(model, X_train, y_train, X_val, y_val, classes):
    """
    Trains the ensemble model and evaluates it on the validation set.

    Args:
        model: The VotingClassifier ensemble to train.
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target labels (integers).
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation target labels (integers).
        classes (np.ndarray): Array of class names (strings) corresponding to integer labels.

    Returns:
        model: The trained model.
    """
    print("Starting training of the Hybrid Kernel-Linear Ensemble...")

    # Fit the ensemble (this fits all sub-estimators)
    model.fit(X_train, y_train)

    print("Training complete. Evaluating on validation set...")

    # Predict probabilities on validation set
    y_pred_proba = model.predict_proba(X_val)

    # Calculate metric
    # We convert integer y_val back to string labels to ensure strict alignment
    # with the classes list in calculate_metric
    if classes is not None:
        y_val_str = classes[y_val]
        score = calculate_metric(y_val_str, y_pred_proba, classes=classes)
    else:
        # Fallback (though classes should be provided)
        score = calculate_metric(y_val, y_pred_proba)

    print(f"Validation Multi-class Log Loss: {score}")

    return model
