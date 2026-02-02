import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from library.config import Config
from library.utils import JigsawMetrics


class RidgeRegressor:
    """
    A wrapper class for Ridge Regression tailored for the Jigsaw Toxicity Classification task.
    It handles training, prediction (with clipping), evaluation using competition metrics,
    and submission file generation.
    """

    def __init__(
        self,
        alpha=Config.RIDGE_ALPHA,
        solver=Config.RIDGE_SOLVER,
        random_state=Config.SEED,
    ):
        """
        Initialize the Ridge Regressor.

        Args:
            alpha (float): Regularization strength.
            solver (str): Solver to use ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga').
            random_state (int): Seed for reproducibility.
        """
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state

        # Initialize the sklearn Ridge model
        self.model = Ridge(
            alpha=self.alpha, solver=self.solver, random_state=self.random_state
        )

    def train(self, X_train, y_train):
        """
        Trains the Ridge Regression model.

        Args:
            X_train (scipy.sparse.csr_matrix): Training feature matrix.
            y_train (array-like): Target values (continuous toxicity scores).
        """
        print(
            f"Training Ridge Regression model (Alpha={self.alpha}, Solver={self.solver})..."
        )
        self.model.fit(X_train, y_train)
        print("Training completed.")

    def predict(self, X):
        """
        Generates predictions for the given feature matrix.
        Predictions are clipped to the [0, 1] range.

        Args:
            X (scipy.sparse.csr_matrix): Feature matrix to predict on.

        Returns:
            np.ndarray: Predicted toxicity scores.
        """
        # Predict continuous scores
        preds = self.model.predict(X)

        # Clip predictions to ensure they fall within valid probability range [0, 1]
        # This is crucial as regression can output values < 0 or > 1
        return np.clip(preds, 0.0, 1.0)

    def evaluate(self, X_val, val_df, target_col="target"):
        """
        Evaluates the model on the validation set using the competition's bias metrics.

        Args:
            X_val (scipy.sparse.csr_matrix): Validation feature matrix.
            val_df (pd.DataFrame): Validation dataframe containing targets and identity columns.
            target_col (str): Name of the target column.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        print("Starting evaluation on validation set...")

        # Generate predictions
        val_preds = self.predict(X_val)

        # Initialize metrics calculator
        metrics_helper = JigsawMetrics()

        # Compute metrics
        # Note: val_df must contain the identity columns (male, female, etc.)
        results = metrics_helper.compute_bias_metrics(
            val_df, val_preds, target_col=target_col
        )

        # Print metrics with full precision
        print("==========================================")
        print("VALIDATION RESULTS")
        print("==========================================")
        print(f"Final Weighted Score: {results['score']}")
        print(f"Overall ROC-AUC:      {results['overall_auc']}")
        print(f"Bias Metrics (Generalized Mean p=-5):")
        print(f"  - Subgroup AUC:     {results['subgroup_auc']}")
        print(f"  - BPSN AUC:         {results['bpsn_auc']}")
        print(f"  - BNSP AUC:         {results['bnsp_auc']}")
        print("==========================================")

        return results

    def save_submission(self, test_ids, test_preds, output_path=Config.SUBMISSION_PATH):
        """
        Saves the test predictions to a CSV file in the required format.

        Args:
            test_ids (array-like): IDs of the test comments.
            test_preds (array-like): Predicted toxicity scores.
            output_path (str): File path to save the submission.
        """
        print(f"Generating submission file at: {output_path}")

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "prediction": test_preds})

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved. Shape: {submission_df.shape}")
