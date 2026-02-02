import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.model_definition import get_logistic_regression_model
from library.data_loader import save_submission
from library.config import Config


def run_cross_validation(X, y, n_splits=5, random_state=Config.SEED, model_params=None):
    """
    Performs Stratified K-Fold Cross Validation to evaluate the model.

    Args:
        X (np.ndarray): Feature matrix for training.
        y (np.ndarray): Target labels.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Seed for reproducibility in splitting.
        model_params (dict, optional): Hyperparameters to override in the model.

    Returns:
        float: The average log loss across all folds.
    """
    print(f"Starting Stratified K-Fold CV with {n_splits} folds...")

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Initialize model with optional parameters
        model = get_logistic_regression_model(params=model_params)

        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Predict probabilities on validation fold
        y_pred_proba = model.predict_proba(X_val_fold)

        # Calculate Log Loss
        loss = log_loss(y_val_fold, y_pred_proba)
        fold_scores.append(loss)

        # Print full precision metric
        print(f"Fold {fold + 1} Log Loss: {loss}")

    mean_loss = np.mean(fold_scores)
    print(f"Average Log Loss: {mean_loss}")

    return mean_loss


def train_and_predict(X_train, y_train, X_test, classes, test_ids, model_params=None):
    """
    Retrains the model on the provided training set and generates predictions for the test set.
    Saves the predictions to the submission file.

    Args:
        X_train (np.ndarray): Full training feature matrix.
        y_train (np.ndarray): Full training labels.
        X_test (np.ndarray): Test feature matrix.
        classes (np.ndarray): Array of class names (strings) corresponding to prediction columns.
        test_ids (pd.Series or list): List or Series of test sample IDs.
        model_params (dict, optional): Hyperparameters to override in the model.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    print("Retraining model on the provided training dataset...")

    # Initialize model
    model = get_logistic_regression_model(params=model_params)

    # Fit on the full training data
    model.fit(X_train, y_train)

    print("Generating predictions on the test set...")
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)

    # Create submission DataFrame
    # Columns must correspond to the classes output by the model (which matches the LabelEncoder order)
    submission_df = pd.DataFrame(y_pred_proba, columns=classes)

    # Insert the 'id' column at the beginning
    submission_df.insert(0, "id", list(test_ids))

    # Save the submission file
    save_submission(submission_df)

    return submission_df
