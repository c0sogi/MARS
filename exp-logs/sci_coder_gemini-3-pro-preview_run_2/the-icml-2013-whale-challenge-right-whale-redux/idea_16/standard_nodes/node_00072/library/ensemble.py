import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import calculate_auc


class MetaLearner:
    """
    Level 1 Meta-Learner for Stacked Generalization.
    Implements a Logistic Regression model to combine predictions from heterogeneous base learners.
    """

    def __init__(self, random_state=Config.SEED, max_iter=1000, C=1.0):
        """
        Args:
            random_state (int): Seed for reproducibility.
            max_iter (int): Maximum number of iterations for the solver.
            C (float): Inverse of regularization strength; smaller values specify stronger regularization.
        """
        self.model = LogisticRegression(
            solver="liblinear",
            random_state=random_state,
            max_iter=max_iter,
            C=C,
        )
        self.is_fitted = False

    def fit(self, oof_preds, targets):
        """
        Trains the meta-learner on Out-Of-Fold (OOF) predictions.

        Args:
            oof_preds (pd.DataFrame or dict): Dictionary or DataFrame where keys/columns are model names
                                              and values are arrays of OOF probabilities.
            targets (np.array or list): Ground truth binary labels corresponding to the OOF predictions.

        Returns:
            float: The AUC score calculated on the OOF dataset.
        """
        # Convert inputs to appropriate formats
        X = pd.DataFrame(oof_preds)
        y = np.array(targets)

        print(f"Training Meta-Learner (Logistic Regression) on {len(X)} samples...")
        print(f"Input Models: {list(X.columns)}")

        self.model.fit(X, y)
        self.is_fitted = True

        # In-sample evaluation (OOF Performance)
        # We use the probability of the positive class (1)
        probs = self.model.predict_proba(X)[:, 1]
        auc = calculate_auc(y, probs)

        print(f"Meta-Learner OOF AUC: {auc:.10f}")
        print(f"Learned Coefficients: {self.model.coef_}")
        print(f"Intercept: {self.model.intercept_}")

        return auc

    def predict(self, test_preds):
        """
        Generates final calibrated probabilities using the fitted meta-learner.

        Args:
            test_preds (pd.DataFrame or dict): Dictionary or DataFrame of test predictions.
                                               Must have the same columns/keys as used in fit().

        Returns:
            np.array: Final predicted probabilities for the positive class.
        """
        if not self.is_fitted:
            raise RuntimeError("MetaLearner must be fitted before prediction.")

        X = pd.DataFrame(test_preds)

        # Predict probabilities for class 1
        return self.model.predict_proba(X)[:, 1]


def aggregate_predictions(pred_list):
    """
    Aggregates predictions across multiple folds or runs (Bagging).
    Used to average the 5-fold test predictions of a single base model
    before passing to the meta-learner.

    Args:
        pred_list (list of np.array): List of prediction arrays (e.g., [fold0_preds, fold1_preds, ...]).

    Returns:
        np.array: The element-wise mean of the input predictions.
    """
    if not pred_list:
        return None

    # Stack and average along the new dimension (axis 0)
    return np.mean(pred_list, axis=0)


def save_submission(clip_names, probabilities, filename="submission.csv"):
    """
    Saves the final predictions to a CSV file in the required format.
    Saves to both the working directory and the ./submission/ directory.

    Args:
        clip_names (list or np.array): List of clip filenames (identifiers).
        probabilities (list or np.array): List of predicted probabilities.
        filename (str): Name of the output file.
    """
    # Create DataFrame matching the submission format
    df = pd.DataFrame({"clip": clip_names, "probability": probabilities})

    # 1. Save to Config.OUTPUT_DIR (Working Directory for persistence/debugging)
    work_path = os.path.join(Config.OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(work_path), exist_ok=True)
    df.to_csv(work_path, index=False)
    print(f"Submission saved to working dir: {work_path}")

    # 2. Save to ./submission/ (Competition Requirement)
    # This is the standard path often used by grading systems
    comp_path = os.path.join("./submission", filename)
    os.makedirs(os.path.dirname(comp_path), exist_ok=True)
    df.to_csv(comp_path, index=False)
    print(f"Submission saved to submission dir: {comp_path}")
