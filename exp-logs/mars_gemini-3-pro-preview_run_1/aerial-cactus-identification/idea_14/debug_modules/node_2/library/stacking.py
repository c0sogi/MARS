import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger(name="stacking")


class StackingEnsemble:
    """
    Level-2 Stacking Ensemble using Logistic Regression.
    Aggregates predictions from diverse base learners to produce a final calibrated probability.
    """

    def __init__(self, random_state=Config.SEED):
        """
        Initialize the meta-learner.

        Args:
            random_state (int): Seed for reproducibility.
        """
        self.random_state = random_state
        # Use liblinear for binary classification with small-medium datasets
        self.model = LogisticRegression(
            random_state=self.random_state, solver="liblinear"
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the meta-learner on OOF predictions.

        Args:
            X (np.ndarray): OOF predictions from base models. Shape (N_samples, N_models).
            y (np.ndarray): Ground truth labels. Shape (N_samples,).
        """
        logger.info(
            f"Fitting Stacking Ensemble on {len(X)} samples with {X.shape[1]} base models..."
        )
        self.model.fit(X, y)
        self.is_fitted = True

        # Log coefficients to understand model contribution
        coefs = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        logger.info(f"Meta-Learner Intercept: {intercept}")
        for i, coef in enumerate(coefs):
            logger.info(f"  Model {i} Coefficient: {coef}")

    def predict(self, X):
        """
        Generate calibrated probabilities.

        Args:
            X (np.ndarray): Predictions from base models. Shape (N_samples, N_models).

        Returns:
            np.ndarray: Final probabilities for the positive class.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # Predict probabilities for class 1
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        """
        Evaluate the model on a given set (usually the OOF set itself to check fit).

        Args:
            X (np.ndarray): Predictions from base models.
            y (np.ndarray): Ground truth labels.

        Returns:
            float: ROC AUC score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        preds = self.predict(X)
        score = roc_auc_score(y, preds)
        return score


def train_meta_learner(oof_preds, y_true):
    """
    Orchestrates the training of the stacking ensemble.

    Args:
        oof_preds (np.ndarray): OOF predictions matrix.
        y_true (np.ndarray): True labels.

    Returns:
        StackingEnsemble: The trained model instance.
        float: The OOF AUC score.
    """
    meta_model = StackingEnsemble(random_state=Config.SEED)

    # Fit model
    meta_model.fit(oof_preds, y_true)

    # Evaluate on OOF data (Training fit)
    oof_auc = meta_model.evaluate(oof_preds, y_true)

    # Print full precision metric
    print(f"Meta-Learner OOF AUC: {oof_auc}")

    return meta_model, oof_auc


def generate_submission_file(
    meta_model, test_preds, test_ids, output_dir=Config.SUBMISSION_DIR
):
    """
    Generates the final submission CSV.

    Args:
        meta_model (StackingEnsemble): The trained meta-learner.
        test_preds (np.ndarray): Test set predictions from base models. Shape (N_test, N_models).
        test_ids (np.ndarray or list): IDs corresponding to the test set images.
        output_dir (str): Directory to save the submission file.
    """
    logger.info("Generating final submission file...")

    # Generate final probabilities
    final_probs = meta_model.predict(test_preds)

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_probs})

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    save_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    logger.info(f"Submission saved successfully to: {save_path}")
    logger.info(f"Submission Head:\n{submission_df.head()}")
