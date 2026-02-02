import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss, accuracy_score
from library import config


def train_classifier(features, labels, load_cached_model=True):
    """
    Trains a LogisticRegressionCV model on the provided features and labels.
    Implements caching to save/load the trained model.

    Args:
        features (np.ndarray): Training features of shape (N, D).
        labels (np.ndarray): Training labels of shape (N,).
        load_cached_model (bool): If True, attempts to load a pre-trained model from disk.

    Returns:
        model (sklearn.linear_model.LogisticRegressionCV): The trained model.
    """
    model_path = config.CACHE_PATHS["model"]

    # Ensure working directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Attempt to load cached model
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached model from {model_path}...")
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print(f"Failed to load cached model: {e}. Retraining...")

    print("Training LogisticRegressionCV classifier...")

    # Initialize LogisticRegressionCV
    # Cs=10 (default) tests 10 values on a log scale
    # cv=5 (from config) performs 5-fold cross-validation
    # solver='lbfgs' (from config) is efficient for multiclass
    model = LogisticRegressionCV(
        Cs=10,
        cv=config.LOGREG_CV,
        solver=config.LOGREG_SOLVER,
        max_iter=config.LOGREG_MAX_ITER,
        n_jobs=config.LOGREG_N_JOBS,
        random_state=config.SEED,
        multi_class="multinomial",  # Explicitly minimize multinomial loss
        verbose=0,
    )

    # Fit the model
    model.fit(features, labels)

    # Save the model
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)

    return model


def evaluate_model(model, features, labels):
    """
    Evaluates the model on validation data using Multi Class Log Loss.

    Args:
        model: Trained sklearn model.
        features (np.ndarray): Validation features.
        labels (np.ndarray): Validation labels.

    Returns:
        float: The calculated log loss.
    """
    print("Evaluating model...")

    # Predict probabilities
    # shape: (N_samples, N_classes)
    probas = model.predict_proba(features)

    # Calculate Log Loss
    # labels are integers, probas are floats
    loss = log_loss(labels, probas)

    # Calculate Accuracy for additional context
    preds = probas.argmax(axis=1)
    acc = accuracy_score(labels, preds)

    print(f"Validation Multi Class Log Loss: {loss}")
    print(f"Validation Accuracy: {acc}")

    return loss


def predict_proba(model, features):
    """
    Generates class probabilities for the given features.

    Args:
        model: Trained sklearn model.
        features (np.ndarray): Test features.

    Returns:
        np.ndarray: Predicted probabilities of shape (N, C).
    """
    return model.predict_proba(features)


def create_submission(ids, probabilities, class_names, output_path=None):
    """
    Creates a submission CSV file in the required format.

    Args:
        ids (np.ndarray): Array of test image IDs.
        probabilities (np.ndarray): Array of predicted probabilities (N, C).
        class_names (list): List of class names corresponding to the probability columns.
        output_path (str, optional): Path to save the CSV. Defaults to config.CACHE_PATHS['submission'].
    """
    if output_path is None:
        output_path = config.CACHE_PATHS["submission"]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating submission file at {output_path}...")

    # Create DataFrame
    # Structure: id, breed1, breed2, ...
    df = pd.DataFrame(probabilities, columns=class_names)
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print("Submission file saved successfully.")
