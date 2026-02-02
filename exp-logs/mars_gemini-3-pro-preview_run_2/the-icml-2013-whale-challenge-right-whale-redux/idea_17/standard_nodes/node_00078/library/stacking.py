import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc


class MetaLearner:
    """
    Wrapper for the Level-1 Logistic Regression Meta-Learner.
    Manages training, prediction, and pickle-free persistence.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        # Initialize Logistic Regression
        # Using 'liblinear' as it is efficient for small feature sets (meta-features)
        self.model = LogisticRegression(
            random_state=Config.SEED, solver="liblinear", class_weight=None
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fits the logistic regression model.

        Args:
            X (np.ndarray): Input features (OOF predictions from base models).
            y (np.ndarray): Target labels.
        """
        print("Fitting Meta-Learner (Logistic Regression)...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate and print training metric for sanity check
        preds = self.model.predict_proba(X)[:, 1]
        auc = calculate_roc_auc(y, preds)
        print(f"Meta-Learner Training AUC: {auc}")
        return auc

    def predict(self, X):
        """
        Predicts probabilities.

        Args:
            X (np.ndarray): Input features (Test predictions from base models).

        Returns:
            np.ndarray: Probabilities for class 1.
        """
        if not self.is_fitted:
            raise ValueError("MetaLearner must be fitted or loaded before prediction.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, directory):
        """
        Saves model coefficients to .npy files (avoiding pickle).
        """
        if not self.is_fitted:
            print("Warning: Model not fitted, cannot save.")
            return

        os.makedirs(directory, exist_ok=True)
        coef_path = os.path.join(directory, "meta_learner_coef.npy")
        intercept_path = os.path.join(directory, "meta_learner_intercept.npy")

        np.save(coef_path, self.model.coef_)
        np.save(intercept_path, self.model.intercept_)
        print(f"Meta-Learner weights saved to {directory}")

    def load(self, directory):
        """
        Loads model coefficients from .npy files.
        """
        coef_path = os.path.join(directory, "meta_learner_coef.npy")
        intercept_path = os.path.join(directory, "meta_learner_intercept.npy")

        if os.path.exists(coef_path) and os.path.exists(intercept_path):
            self.model.coef_ = np.load(coef_path)
            self.model.intercept_ = np.load(intercept_path)
            # Manually set classes_ as we are bypassing fit()
            self.model.classes_ = np.array([0, 1])
            self.is_fitted = True
            print(f"Meta-Learner weights loaded from {directory}")
            return True
        else:
            print(f"No Meta-Learner weights found at {directory}")
            return False


def fit_stacking_model(X, y, save_dir=None):
    """
    Trains the stacking meta-learner.

    Args:
        X (np.ndarray): Matrix of OOF predictions (N_samples, N_models).
        y (np.ndarray): Ground truth labels.
        save_dir (str, optional): Directory to save model weights.

    Returns:
        MetaLearner: The trained model instance.
    """
    learner = MetaLearner()
    learner.fit(X, y)

    if save_dir:
        learner.save(save_dir)

    return learner


def predict_stacking_model(model, X):
    """
    Generates predictions using the meta-learner.

    Args:
        model (MetaLearner): Trained model instance.
        X (np.ndarray): Matrix of test predictions (N_samples, N_models).

    Returns:
        np.ndarray: Final probabilities.
    """
    return model.predict(X)
