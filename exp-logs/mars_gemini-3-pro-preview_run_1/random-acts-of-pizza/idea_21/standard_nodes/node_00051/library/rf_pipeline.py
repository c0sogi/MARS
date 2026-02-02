import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed


def train_rf_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_estimators=Config.RF_N_ESTIMATORS,
    max_depth=Config.RF_MAX_DEPTH,
    class_weight=Config.RF_CLASS_WEIGHT,
    n_jobs=Config.RF_N_JOBS,
    random_state=Config.RANDOM_SEED,
):
    """
    Trains a Random Forest Classifier on the provided features.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        X_val (np.array, optional): Validation features for metric evaluation.
        y_val (np.array, optional): Validation targets for metric evaluation.
        n_estimators (int): Number of trees in the forest.
        max_depth (int or None): Maximum depth of the tree.
        class_weight (str or dict): Weights associated with classes.
        n_jobs (int): Number of jobs to run in parallel.
        random_state (int): Seed used by the random number generator.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained model.
    """
    # Ensure reproducibility
    set_seed(random_state)

    print(
        f"Initializing Random Forest (n_estimators={n_estimators}, max_depth={max_depth})..."
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )

    print("Fitting Random Forest model...")
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        print("Evaluating Random Forest on validation set...")
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        print("Validation ROC AUC:")
        print(val_auc)

    return model


def predict_rf(model, X):
    """
    Generates probability predictions for the positive class.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained model.
        X (np.array): Features to predict on.

    Returns:
        np.array: Predicted probabilities for class 1.
    """
    # Return probabilities for the positive class (index 1)
    return model.predict_proba(X)[:, 1]
