import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from library.config import Config
from library.utils import compute_rmse


class RidgeHead:
    """
    Linear probing head using Ridge Regression.
    Concatenates image embeddings and metadata, scales features, and predicts Pawpularity.
    """

    def __init__(self, alpha=Config.RIDGE_ALPHA):
        """
        Args:
            alpha (float): Regularization strength for Ridge Regression.
        """
        self.alpha = alpha
        # Pipeline:
        # 1. StandardScaler: Standardize features by removing the mean and scaling to unit variance.
        #    This is essential for Ridge regression to treat all features (embeddings + metadata) equally.
        # 2. Ridge: Linear least squares with L2 regularization.
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    Ridge(
                        alpha=self.alpha,
                        random_state=Config.SEED,
                        solver="auto",  # Let sklearn choose the most efficient solver (e.g., Cholesky)
                    ),
                ),
            ]
        )

    def _prepare_X(self, embeddings, metadata):
        """
        Combines image embeddings and metadata into a single feature matrix.

        Args:
            embeddings (np.ndarray): Shape (N, D)
            metadata (np.ndarray): Shape (N, M)

        Returns:
            np.ndarray: Shape (N, D + M)
        """
        # Ensure inputs are numpy arrays
        embeddings = np.array(embeddings)
        metadata = np.array(metadata)

        # Concatenate along the feature axis
        if metadata.size == 0:
            return embeddings
        return np.hstack([embeddings, metadata])

    def fit(self, embeddings, metadata, targets):
        """
        Trains the Ridge regression model.

        Args:
            embeddings (np.ndarray): Training image features.
            metadata (np.ndarray): Training metadata features.
            targets (np.ndarray): Training target values.
        """
        X = self._prepare_X(embeddings, metadata)
        y = np.array(targets)

        print(
            f"Training Ridge Regressor (alpha={self.alpha}) on {X.shape[0]} samples with {X.shape[1]} features..."
        )
        self.model.fit(X, y)

        # Calculate and print training metric
        train_preds = self.model.predict(X)
        rmse = compute_rmse(y, train_preds)
        print(f"Training RMSE: {rmse}")

    def predict(self, embeddings, metadata):
        """
        Generates predictions using the trained model.

        Args:
            embeddings (np.ndarray): Image features.
            metadata (np.ndarray): Metadata features.

        Returns:
            np.ndarray: Predicted Pawpularity scores.
        """
        X = self._prepare_X(embeddings, metadata)
        return self.model.predict(X)

    def evaluate(self, embeddings, metadata, targets):
        """
        Evaluates the model on a validation set.

        Args:
            embeddings (np.ndarray): Validation image features.
            metadata (np.ndarray): Validation metadata features.
            targets (np.ndarray): Validation target values.

        Returns:
            float: The calculated RMSE.
        """
        preds = self.predict(embeddings, metadata)
        rmse = compute_rmse(targets, preds)
        # Print full precision as requested
        print(f"Validation RMSE: {rmse}")
        return rmse

    def save_submission(
        self, test_ids, predictions, output_path=Config.SUBMISSION_PATH
    ):
        """
        Saves predictions to a CSV file in the required format.

        Args:
            test_ids (np.ndarray or list): List of Pet Profile IDs.
            predictions (np.ndarray or list): List of predicted scores.
            output_path (str): Path to save the submission file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"Id": test_ids, "Pawpularity": np.array(predictions).flatten()}
        )

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
