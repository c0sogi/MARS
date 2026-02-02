import numpy as np
import pandas as pd
import os
from library.config import SUBMISSION_FILE_PATH


class SelectiveEnsemble:
    """
    Manages a pool of candidate models trained on different views, applies
    performance-based filtering, and generates ensemble predictions.
    """

    def __init__(self, tolerance=0.05):
        """
        Args:
            tolerance (float): The maximum allowed deviation in Log Loss from the
                               Global Baseline for a model to be included.
        """
        self.tolerance = tolerance
        self.candidates = []
        # Initialize with -inf because metric is neg_log_loss (higher is better, e.g. -0.1 > -0.5)
        self.global_best_score = -np.inf
        self.selected_candidates = []

    def add_candidate(self, model, view_name, score, model_name):
        """
        Registers a trained model candidate.

        Args:
            model: The trained scikit-learn estimator.
            view_name (str): The feature view used ('Global', 'Margin', 'Shape', 'Texture').
            score (float): The cross-validation score (neg_log_loss).
            model_name (str): A descriptive identifier for the model.
        """
        candidate = {
            "model": model,
            "view": view_name,
            "score": score,
            "name": model_name,
        }
        self.candidates.append(candidate)

        # Update the Global Baseline if this is a Global view model
        # We track the maximum (least negative) score among Global models
        if view_name == "Global":
            if score > self.global_best_score:
                self.global_best_score = score

    def optimize_selection(self):
        """
        Selects all candidates for Soft Voting.
        Pruning based on individual performance is disabled to preserve ensemble diversity.
        Cite solution_lesson_node_00036
        """
        print(
            f"Ensemble Selection: Selecting all {len(self.candidates)} candidates for Soft Voting."
        )
        self.selected_candidates = self.candidates

        for cand in self.selected_candidates:
            print(
                f"  [SELECTED] {cand['name']:<25} | View: {cand['view']:<8} | Score: {cand['score']:.6f}"
            )

    def predict(self, X_test_views):
        """
        Generates averaged probability predictions from selected models.

        Args:
            X_test_views (dict): Dictionary mapping view names to test feature arrays.

        Returns:
            np.ndarray: Averaged probability matrix (n_samples, n_classes).
        """
        if not self.selected_candidates:
            raise ValueError(
                "No candidates selected. Call optimize_selection() before predicting."
            )

        predictions_list = []

        for cand in self.selected_candidates:
            view_name = cand["view"]
            model = cand["model"]

            if view_name not in X_test_views:
                raise KeyError(
                    f"View '{view_name}' required by {cand['name']} not found in test data."
                )

            X_target = X_test_views[view_name]

            # Generate probabilities
            # Output shape: (n_samples, n_classes)
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
