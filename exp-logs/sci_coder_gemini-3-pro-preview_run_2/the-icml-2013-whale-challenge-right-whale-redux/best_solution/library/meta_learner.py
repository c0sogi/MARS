import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything


class StackingEnsemble:
    """
    Level 1 Meta-Learner for the Stacked Generalization Ensemble.
    Uses Logistic Regression to combine predictions from base learners.
    """

    def __init__(self):
        """
        Initialize the StackingEnsemble with a Logistic Regression model.
        Sets random seeds for reproducibility.
        """
        seed_everything(Config.SEED)

        # Initialize Logistic Regression
        # Solver 'liblinear' is good for smaller datasets and binary classification.
        # No regularization penalty or weak regularization (C=1.0) usually works well for stacking
        # as features (probabilities) are already highly correlated with the target.
        self.model = LogisticRegression(
            solver="liblinear", random_state=Config.SEED, C=1.0, fit_intercept=True
        )
        self.is_fitted = False

    def fit(self, X_oof, y_oof):
        """
        Trains the meta-learner on Out-Of-Fold predictions.

        Args:
            X_oof (np.array): OOF predictions from base learners. Shape (n_samples, n_models).
            y_oof (np.array): Ground truth labels. Shape (n_samples,).
        """
        print("Training Meta-Learner (Logistic Regression)...")

        # Handle potential NaNs by replacing them with 0.5 (neutral probability)
        # This ensures robustness if a fold failed for some reason.
        if np.isnan(X_oof).any():
            print("Warning: NaNs detected in OOF features. Replacing with 0.5.")
            X_oof = np.nan_to_num(X_oof, nan=0.5)

        self.model.fit(X_oof, y_oof)
        self.is_fitted = True

        # Evaluate on the training set (which is the OOF set)
        # This gives an unbiased estimate of the ensemble's performance.
        oof_probs = self.model.predict_proba(X_oof)[:, 1]
        oof_auc = roc_auc_score(y_oof, oof_probs)

        print(f"Meta-Learner OOF AUC: {oof_auc}")
        print(f"Meta-Learner Coefficients: {self.model.coef_}")
        print(f"Meta-Learner Intercept: {self.model.intercept_}")

    def predict(self, X_test):
        """
        Generates predictions for the test set using the trained meta-learner.

        Args:
            X_test (np.array): Test set predictions from base learners. Shape (n_samples, n_models).

        Returns:
            np.array: Predicted probabilities for the positive class.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        # Handle NaNs
        if np.isnan(X_test).any():
            print("Warning: NaNs detected in Test features. Replacing with 0.5.")
            X_test = np.nan_to_num(X_test, nan=0.5)

        return self.model.predict_proba(X_test)[:, 1]

    def create_submission(self, test_probs, test_clips):
        """
        Creates and saves the submission file.

        Args:
            test_probs (np.array): Predicted probabilities for the test set.
            test_clips (list or pd.Series): Filenames for the test clips.
        """
        print("Generating submission file...")

        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Create DataFrame matching the sampleSubmission format
        submission = pd.DataFrame({"clip": test_clips, "probability": test_probs})

        # Save to CSV
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(save_path, index=False)

        print(f"Submission saved successfully to {save_path}")
        print("Head of submission:")
        print(submission.head())
