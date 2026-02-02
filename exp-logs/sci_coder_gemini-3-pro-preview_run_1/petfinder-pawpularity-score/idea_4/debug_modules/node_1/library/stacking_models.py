import os
import numpy as np
import joblib
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List

from library.config import Config
from library.utils import seed_everything, compute_rmse


class Level0Expert:
    """
    Level 0 Expert Model for the Stacking Ensemble.

    This model specializes in a specific image backbone (e.g., Swin, CLIP).
    It fuses the extracted image embeddings with the binary metadata features,
    scales the combined vector, and fits a Ridge Regression model with
    built-in hyperparameter optimization (RidgeCV).
    """

    def __init__(self, alphas: Tuple[float, ...] = Config.RIDGE_ALPHAS):
        """
        Args:
            alphas (Tuple[float, ...]): List of alpha values for RidgeCV to try.
        """
        seed_everything(Config.SEED)
        self.alphas = alphas
        self.scaler = StandardScaler()
        # cv=None uses efficient Leave-One-Out Cross-Validation for Ridge
        self.model = RidgeCV(alphas=self.alphas, cv=None, scoring=None)
        self.is_fitted = False

    def _prepare_data(
        self, img_features: np.ndarray, meta_features: np.ndarray
    ) -> np.ndarray:
        """
        Concatenates image features and metadata features.

        Args:
            img_features (np.ndarray): Shape (N, D_img)
            meta_features (np.ndarray): Shape (N, D_meta)

        Returns:
            np.ndarray: Concatenated features (N, D_img + D_meta)
        """
        # Ensure inputs are 2D
        if img_features.ndim == 1:
            img_features = img_features.reshape(1, -1)
        if meta_features.ndim == 1:
            meta_features = meta_features.reshape(1, -1)

        # Concatenate along the feature dimension
        return np.concatenate([img_features, meta_features], axis=1)

    def fit(
        self, img_features: np.ndarray, meta_features: np.ndarray, targets: np.ndarray
    ) -> None:
        """
        Fits the scaler and the RidgeCV model.

        Args:
            img_features (np.ndarray): Image embeddings.
            meta_features (np.ndarray): Metadata flags.
            targets (np.ndarray): Target Pawpularity scores.
        """
        X = self._prepare_data(img_features, meta_features)
        y = targets.ravel()  # Ensure y is (N,)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Log training info
        print(f"  [Level0Expert] Fitted. Best Alpha: {self.model.alpha_}")
        print(
            f"  [Level0Expert] R^2 Score on training data: {self.model.score(X_scaled, y)}"
        )

    def predict(
        self, img_features: np.ndarray, meta_features: np.ndarray
    ) -> np.ndarray:
        """
        Predicts Pawpularity scores.

        Args:
            img_features (np.ndarray): Image embeddings.
            meta_features (np.ndarray): Metadata flags.

        Returns:
            np.ndarray: Predicted scores (N,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        X = self._prepare_data(img_features, meta_features)
        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled)

        # Clip predictions to valid range [0, 100] as Pawpularity is strictly bounded
        # Although Ridge is linear, clipping is a safe post-processing step.
        preds = np.clip(preds, 0.0, 100.0)

        return preds

    def save(self, filepath: str) -> None:
        """Saves the fitted scaler and model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        dump_data = {"scaler": self.scaler, "model": self.model, "alphas": self.alphas}
        joblib.dump(dump_data, filepath)
        print(f"  [Level0Expert] Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads the scaler and model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.alphas = data.get("alphas", self.alphas)
        self.is_fitted = True
        print(f"  [Level0Expert] Model loaded from {filepath}")


class Level1MetaLearner:
    """
    Level 1 Meta-Learner for the Stacking Ensemble.

    This model takes the predictions from multiple Level 0 Experts and learns
    an optimal weighted combination. It uses Ridge Regression with a positivity
    constraint to ensure the ensemble acts as a weighted average.
    """

    def __init__(self, alpha: float = 1.0, positive: bool = True):
        """
        Args:
            alpha (float): Regularization strength.
            positive (bool): Whether to force coefficients to be positive.
        """
        seed_everything(Config.SEED)
        # Using standard Ridge. Positive=True is available in sklearn >= 1.0
        # It forces coefficients to be >= 0, which is beneficial for ensembling.
        self.model = Ridge(
            alpha=alpha, fit_intercept=True, positive=positive, random_state=Config.SEED
        )
        self.is_fitted = False

    def fit(self, expert_preds: np.ndarray, targets: np.ndarray) -> None:
        """
        Fits the meta-learner.

        Args:
            expert_preds (np.ndarray): Matrix of predictions from experts (N_samples, N_experts).
            targets (np.ndarray): True target values (N_samples,).
        """
        y = targets.ravel()

        self.model.fit(expert_preds, y)
        self.is_fitted = True

        print(f"  [Level1MetaLearner] Fitted.")
        print(f"  [Level1MetaLearner] Coefficients (Weights): {self.model.coef_}")
        print(f"  [Level1MetaLearner] Intercept: {self.model.intercept_}")

    def predict(self, expert_preds: np.ndarray) -> np.ndarray:
        """
        Generates final ensemble predictions.

        Args:
            expert_preds (np.ndarray): Matrix of predictions from experts (N_samples, N_experts).

        Returns:
            np.ndarray: Final predictions (N_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner must be fitted before calling predict.")

        preds = self.model.predict(expert_preds)

        # Clip final predictions to valid range
        preds = np.clip(preds, 0.0, 100.0)

        return preds

    def save(self, filepath: str) -> None:
        """Saves the meta-learner to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"  [Level1MetaLearner] Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads the meta-learner from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"  [Level1MetaLearner] Model loaded from {filepath}")
