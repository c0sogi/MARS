import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_rf(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators=Config.RF_ESTIMATORS,
    min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
    class_weight=Config.RF_CLASS_WEIGHT,
    random_state=Config.SEED,
):
    """
    Trains a Random Forest classifier with the specified hyperparameters.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        n_estimators (int): Number of trees in the forest.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        class_weight (str or dict): Weights associated with classes.
        random_state (int): Seed used by the random number generator.

    Returns:
        model: The trained RandomForestClassifier.
    """

    # Initialize the classifier with provided hyperparameters
    # n_jobs=-1 utilizes all available vCPUs
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Predict probabilities on the validation set
    # We take the probability of the positive class (index 1)
    val_probs = clf.predict_proba(X_val)[:, 1]

    # Calculate AUC score
    val_auc = roc_auc_score(y_val, val_probs)

    # Print the validation AUC with full precision
    print(f"RF Validation AUC: {val_auc}")

    return clf


def predict_rf(model, X_test):
    """
    Generates probability predictions using the trained model.

    Args:
        model: Trained RandomForestClassifier.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: Predicted probabilities for the positive class.
    """
    # Predict probabilities for the positive class
    probs = model.predict_proba(X_test)[:, 1]
    return probs
