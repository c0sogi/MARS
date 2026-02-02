import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def calculate_log_loss(y_true, y_pred_proba, classes):
    """
    Computes the multi-class log loss on the validation set.

    Args:
        y_true (np.ndarray): True class labels (strings).
        y_pred_proba (np.ndarray): Predicted probabilities from the model.
        classes (np.ndarray): Array of class names corresponding to the columns of y_pred_proba.

    Returns:
        float: The calculated log loss.
    """
    # Calculate multi-class log loss
    # We provide 'labels' to ensure sklearn maps the columns of probabilities
    # to the correct string labels in y_true.
    loss = log_loss(y_true, y_pred_proba, labels=classes)

    # Print full precision without rounding
    print(f"Validation Multi-class Log Loss: {loss}")

    return loss


def create_submission_file(model, X_test, test_ids, output_path=Config.SUBMISSION_FILE):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained model instance (must have predict_proba and classes_).
        X_test (np.ndarray): Test features.
        test_ids (np.ndarray): Test image IDs.
        output_path (str): File path to save the submission CSV.
    """
    print("Generating predictions for test set...")

    # Generate probabilities
    # The model's predict_proba is expected to handle any necessary clipping/normalization
    y_pred_proba = model.predict_proba(X_test)

    # Retrieve class names from the model to use as column headers
    classes = model.classes_

    # Create DataFrame
    df_submission = pd.DataFrame(y_pred_proba, columns=classes)

    # Insert ID column at the beginning
    df_submission.insert(0, "id", test_ids)

    # Ensure IDs are integers
    df_submission["id"] = df_submission["id"].astype(int)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    print(f"Saving submission to {output_path}...")
    df_submission.to_csv(output_path, index=False)

    print("Submission generated successfully.")
