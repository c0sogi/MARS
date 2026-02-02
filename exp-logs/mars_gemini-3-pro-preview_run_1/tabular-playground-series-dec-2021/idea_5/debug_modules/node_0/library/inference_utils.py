import os
import numpy as np
import pandas as pd


def soft_vote_predict(models, X_test):
    """
    Aggregates predictions from a list of trained models using soft voting.

    Args:
        models (list): List of trained classifier objects (e.g., XGBClassifier).
        X_test (pd.DataFrame or np.ndarray): Test features.

    Returns:
        np.ndarray: Array of predicted class labels.
    """
    if not models:
        raise ValueError("No models provided for prediction.")

    print(f"Aggregating predictions from {len(models)} models...")

    # Initialize accumulator for probabilities
    sum_probs = None

    for i, model in enumerate(models):
        # Predict probabilities
        # XGBoost predict_proba returns a numpy array of shape (n_samples, n_classes)
        probs = model.predict_proba(X_test)

        if sum_probs is None:
            sum_probs = np.zeros_like(probs)

        sum_probs += probs

    # Compute average probabilities (soft voting)
    # Note: argmax on sum is equivalent to argmax on average, but average is conceptually cleaner
    avg_probs = sum_probs / len(models)

    # Select class with highest probability
    # Based on the configuration (num_class=8) and data (labels 1-7),
    # the array indices 0-7 correspond directly to the class labels.
    predictions = np.argmax(avg_probs, axis=1)

    return predictions


def export_submission(ids, predictions, output_path):
    """
    Formats the predictions and saves them to a CSV file.

    Args:
        ids (pd.Series or np.ndarray): The ID column for the test set.
        predictions (np.ndarray): The predicted class labels.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame matching the submission format
    submission = pd.DataFrame({"Id": ids, "Cover_Type": predictions})

    # Save to CSV without the index
    print(f"Saving submission to {output_path}...")
    submission.to_csv(output_path, index=False)
    print("Submission saved successfully.")
