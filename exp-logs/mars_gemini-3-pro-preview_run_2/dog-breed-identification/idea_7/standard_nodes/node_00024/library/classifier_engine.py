import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from library.config import Config


def train_and_evaluate(train_features, train_labels, val_features, val_labels):
    """
    Trains a Logistic Regression classifier with Cross-Validation and evaluates it.

    Args:
        train_features (np.ndarray): Training feature matrix (N_train, D).
        train_labels (np.ndarray): Training labels (N_train,).
        val_features (np.ndarray): Validation feature matrix (N_val, D).
        val_labels (np.ndarray): Validation labels (N_val,).

    Returns:
        model: The trained scikit-learn model.
        metric (float): The validation Log Loss.
    """
    print("Initializing LogisticRegressionCV...")

    # Initialize model with Config parameters
    # Cs=10 is standard for log-spaced grid of C values
    model = LogisticRegressionCV(
        Cs=10,
        cv=Config.CV_FOLDS,
        solver=Config.LOGREG_SOLVER,
        multi_class="multinomial",
        max_iter=Config.LOGREG_MAX_ITER,
        random_state=Config.SEED,
        n_jobs=Config.n_jobs,
        verbose=0,
    )

    print(
        f"Training model on {train_features.shape[0]} samples with {train_features.shape[1]} features..."
    )
    model.fit(train_features, train_labels)

    print("Evaluating on validation set...")
    val_probs = model.predict_proba(val_features)

    # Calculate Log Loss
    # labels are integers, predict_proba returns matrix of shape (n_samples, n_classes)
    metric = log_loss(val_labels, val_probs)

    print(f"Validation Multi Class Log Loss: {metric}")

    return model, metric


def predict_probas(model, features):
    """
    Generates probability predictions for the given features.

    Args:
        model: Trained scikit-learn model.
        features (np.ndarray): Feature matrix (N, D).

    Returns:
        np.ndarray: Probability matrix (N, C).
    """
    return model.predict_proba(features)


def save_model(model, path):
    """
    Saves the trained model to disk.

    Args:
        model: The model object.
        path (str): File path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    Loads a model from disk.

    Args:
        path (str): File path.

    Returns:
        model: The loaded model object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def generate_submission(model, test_features, test_ids, output_path):
    """
    Generates the submission CSV file.

    Args:
        model: Trained model.
        test_features (np.ndarray): Test set features.
        test_ids (np.ndarray): Test set IDs.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating submission file...")

    # 1. Get Predictions
    probs = predict_probas(model, test_features)

    # 2. Get Class Names (Breeds)
    # The model was trained on integer labels mapped from sorted breed names.
    # We need to reconstruct the list of breeds to create the header.
    # We read the training metadata to get the unique breeds.
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    classes = sorted(train_df["breed"].unique())

    # Verify consistency
    if len(classes) != probs.shape[1]:
        raise ValueError(
            f"Number of classes in metadata ({len(classes)}) does not match model output ({probs.shape[1]})."
        )

    # 3. Create DataFrame
    # Columns: id, breed_1, breed_2, ...
    submission_df = pd.DataFrame(probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
