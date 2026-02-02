import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from library.config import Config


def train_stream_classifier(embeddings, labels, stream_name, load_cached_model=True):
    """
    Trains a LogisticRegressionCV classifier on the provided embeddings.
    Implements caching to avoid retraining if the model already exists.

    Args:
        embeddings (np.ndarray): Training features of shape (N, D).
        labels (np.ndarray): Training labels of shape (N,).
        stream_name (str): Unique identifier for the stream (e.g., 'stream_a_convnext').
        load_cached_model (bool): If True, attempts to load a pre-trained model from disk.

    Returns:
        sklearn.linear_model.LogisticRegressionCV: The trained classifier.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    model_filename = f"{stream_name}_logreg_model.joblib"
    model_path = os.path.join(Config.WORKING_DIR, model_filename)

    # 1. Try loading from cache
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached model for {stream_name} from {model_path}...")
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print(f"Failed to load cached model: {e}. Retraining...")

    # 2. Train model
    print(f"Training classifier for {stream_name}...")

    # Set seed for reproducibility
    np.random.seed(Config.SEED)

    # Initialize LogisticRegressionCV
    # - Uses Stratified K-Fold CV by default.
    # - 'multinomial' ensures we minimize the correct loss function.
    # - 'neg_log_loss' scoring guides the CV to pick the C that minimizes log loss.
    clf = LogisticRegressionCV(
        Cs=Config.LOGREG_PARAMS["Cs"],
        cv=Config.LOGREG_PARAMS["cv"],
        solver=Config.LOGREG_PARAMS["solver"],
        max_iter=Config.LOGREG_PARAMS["max_iter"],
        n_jobs=Config.LOGREG_PARAMS["n_jobs"],
        random_state=Config.LOGREG_PARAMS["random_state"],
        multi_class="multinomial",
        scoring="neg_log_loss",
        verbose=0,
    )

    clf.fit(embeddings, labels)

    # Log the average best C to give insight into regularization strength
    avg_c = np.mean(clf.C_)
    print(f"Model trained. Average Best C: {avg_c}")

    # 3. Save to cache
    print(f"Saving model for {stream_name} to {model_path}...")
    joblib.dump(clf, model_path)

    return clf


def predict_stream(model, embeddings):
    """
    Generates probability predictions using the trained model.

    Args:
        model: Trained sklearn model.
        embeddings (np.ndarray): Features of shape (N, D).

    Returns:
        np.ndarray: Predicted probabilities of shape (N, n_classes).
    """
    return model.predict_proba(embeddings)


def evaluate_model(probs, labels):
    """
    Calculates and prints the Multi-Class Log Loss.

    Args:
        probs (np.ndarray): Predicted probabilities.
        labels (np.ndarray): True label indices.

    Returns:
        float: The calculated Log Loss.
    """
    loss = log_loss(labels, probs)
    # Print full precision as requested
    print(f"Validation Log Loss: {loss}")
    return loss


def optimize_ensemble_weights(probs_a, probs_b, labels):
    """
    Finds the optimal scalar weight w for the ensemble: P = w * P_a + (1 - w) * P_b
    Minimizes Log Loss on the validation set.

    Args:
        probs_a (np.ndarray): Probabilities from Stream A.
        probs_b (np.ndarray): Probabilities from Stream B.
        labels (np.ndarray): True label indices.

    Returns:
        float: The optimal weight for Stream A (w).
    """
    print("Optimizing ensemble weights...")

    best_loss = float("inf")
    best_w = 0.5

    # Grid search from 0.0 to 1.0
    # 101 steps allows for 0.00, 0.01, ..., 1.00 resolution
    weights = np.linspace(0, 1, 101)

    for w in weights:
        # Calculate weighted average
        probs_ensemble = w * probs_a + (1.0 - w) * probs_b

        # Calculate loss
        current_loss = log_loss(labels, probs_ensemble)

        if current_loss < best_loss:
            best_loss = current_loss
            best_w = w

    print(f"Optimal Ensemble Weight (Stream A): {best_w}")
    print(f"Best Combined Validation Log Loss: {best_loss}")

    return best_w


def create_submission(ids, probs, output_path=Config.SUBMISSION_PATH):
    """
    Creates and saves the submission CSV file.

    Args:
        ids (np.ndarray): Array of test image IDs.
        probs (np.ndarray): Array of predicted probabilities for the test set.
        output_path (str): File path to save the submission.
    """
    # We need the list of breed names to create the CSV header.
    # The model classes correspond to indices 0..N-1, which are assigned
    # based on the sorted unique breed names in the training set.
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    breeds = sorted(df_train["breed"].unique())

    # Validate shape
    if probs.shape[1] != len(breeds):
        raise ValueError(
            f"Number of predicted classes ({probs.shape[1]}) does not match number of breeds ({len(breeds)})"
        )

    # Create DataFrame
    submission_df = pd.DataFrame(probs, columns=breeds)
    submission_df.insert(0, "id", ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)

    return submission_df
