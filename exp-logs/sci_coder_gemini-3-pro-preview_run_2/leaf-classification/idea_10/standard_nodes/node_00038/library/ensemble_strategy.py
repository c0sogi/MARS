import numpy as np
import pandas as pd
import os
from library.config import SUBMISSION_FILE_PATH


class SoftVotingEnsemble:
    """
    Manages a pool of models and generates ensemble predictions via Soft Voting (Averaging).
    Prioritizes diversity over individual strength (Cite Lesson 36).
    """

    def __init__(self):
        self.models = []

    def add_model(self, model, view_name, name):
        """
        Registers a trained model.

        Args:
            model: The trained scikit-learn estimator.
            view_name (str): The feature view used.
            name (str): A descriptive identifier for the model.
        """
        self.models.append({"model": model, "view": view_name, "name": name})
        print(f"Added {name} to ensemble.")

    def predict(self, X_test_views):
        """
        Generates averaged probability predictions from all models.

        Args:
            X_test_views (dict): Dictionary mapping view names to test feature arrays.

        Returns:
            np.ndarray: Averaged probability matrix (n_samples, n_classes).
        """
        if not self.models:
            raise ValueError("No models in ensemble.")

        predictions_list = []

        for item in self.models:
            view_name = item["view"]
            model = item["model"]

            if view_name not in X_test_views:
                raise KeyError(
                    f"View '{view_name}' required by {item['name']} not found."
                )

            X_target = X_test_views[view_name]
            probs = model.predict_proba(X_target)
            predictions_list.append(probs)

        # Soft Voting: Average the probabilities
        ensemble_prediction = np.mean(predictions_list, axis=0)

        return ensemble_prediction


def generate_submission(
    predictions, test_ids, classes, output_path=SUBMISSION_FILE_PATH
):
    """
    Formats and saves the submission file.

    Args:
        predictions (np.ndarray): The probability predictions (n_samples, n_classes).
        test_ids (np.ndarray): The IDs of the test samples.
        classes (np.ndarray): The class names corresponding to prediction columns.
        output_path (str): File path to save the CSV.
    """
    # Create DataFrame with class columns
    df_submission = pd.DataFrame(predictions, columns=classes)

    # Insert 'id' as the first column
    df_submission.insert(0, "id", test_ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved successfully to: {output_path}")
