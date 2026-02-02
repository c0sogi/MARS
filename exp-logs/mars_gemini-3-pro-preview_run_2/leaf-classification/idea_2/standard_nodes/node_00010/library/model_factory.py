import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier

from library.config import Config
from library.utils import calculate_metric


def create_hybrid_ensemble():
    """
    Constructs the Hybrid Linear Ensemble consisting of:
    1. Logistic Regression (Discriminative Linear)
    2. LDA (Generative Linear)

    Cite solution_lesson_node_00009: Removed SVM to prevent ensemble dilution.
    Cite solution_lesson_node_00006: Generative-Discriminative Ensembling.

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

    # 3. Ensemble Construction
    print("Constructing VotingClassifier...")
    estimators = [("lr", lr_clf), ("lda", lda_clf)]

    # Extract weights ensuring correct order matching estimators list
    weights = [
        Config.ENSEMBLE_WEIGHTS["lr"],
        Config.ENSEMBLE_WEIGHTS["lda"],
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
