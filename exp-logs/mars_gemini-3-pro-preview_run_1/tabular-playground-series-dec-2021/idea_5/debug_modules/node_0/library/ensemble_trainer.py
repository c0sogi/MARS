import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from library.config import SEED, MAX_ROUNDS, EARLY_STOPPING_ROUNDS


def train_ensemble(X, y, params, n_folds=5):
    """
    Trains a homogeneous ensemble of XGBoost models using Stratified K-Fold Cross-Validation.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training targets.
        params (dict): Hyperparameters for the XGBoost classifier.
        n_folds (int): Number of folds for cross-validation.

    Returns:
        tuple: (models, scores)
            - models (list): List of trained XGBClassifier instances.
            - scores (list): List of validation accuracy scores for each fold.
    """
    print(f"Starting Stratified {n_folds}-Fold Ensemble Training...")

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    models = []
    scores = []

    # Ensure inputs are in a format compatible with iloc if they aren't already
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)

    fold_idx = 1

    for train_index, val_index in skf.split(X, y):
        print(f"\n--- Training Fold {fold_idx}/{n_folds} ---")

        # Split data
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize model with provided parameters
        # We override n_estimators with MAX_ROUNDS to control the upper limit of iterations
        # Early stopping will determine the actual number of trees
        clf = XGBClassifier(n_estimators=MAX_ROUNDS, **params)

        # Train the model
        # Note: XGBoost's sklearn API uses fit() with eval_set for early stopping
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

        # Validate
        # We use the best iteration automatically restored by XGBoost
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        print(f"Fold {fold_idx} Accuracy: {acc}")

        # Store model and score
        models.append(clf)
        scores.append(acc)

        fold_idx += 1

    # Calculate and print overall average accuracy
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"\nEnsemble Training Completed.")
    print(f"Average OOF Accuracy: {mean_acc}")
    print(f"Accuracy Std Dev: {std_acc}")

    return models, scores
