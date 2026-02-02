import numpy as np
import pandas as pd
import os
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from library.feature_extractor import DualBackboneExtractor
from library.utils import SUBMISSION_PATH, SEED, set_seed

# Set seed for reproducibility
set_seed(SEED)


class RidgeRegressor:
    """
    A regression model using Ridge Regression with built-in Cross-Validation
    and feature scaling. Designed for high-dimensional feature spaces.
    """

    def __init__(self, alphas=None, cv=5):
        """
        Initialize the regressor.

        Args:
            alphas (np.ndarray, optional): Array of alpha values to try.
                                           If None, uses a broad default range.
            cv (int): Number of cross-validation folds.
        """
        if alphas is None:
            # Define a broad search grid for regularization strength.
            # Expanded range up to 1,000,000 (10^6) to handle increased dimensionality (~4100 features)
            # Cite Lesson 00010: Hyperparameter Search Boundary Expansion
            self.alphas = np.concatenate(
                [np.array([0.1, 1.0, 10.0]), np.logspace(2, 6.0, 50)]
            )
        else:
            self.alphas = alphas

        self.cv = cv
        self.pipeline = None

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Args:
            X (np.ndarray): Training features (samples x features).
            y (np.ndarray): Training targets.
        """
        # Create a pipeline:
        # 1. StandardScaler: Crucial for mixing binary metadata with float embeddings.
        # 2. RidgeCV: Optimizes alpha using efficient LOO-CV or K-Fold.
        self.pipeline = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=self.alphas, cv=self.cv, scoring="neg_mean_squared_error"),
        )
        self.pipeline.fit(X, y)

    def predict(self, X):
        """
        Predict target values for new data.

        Args:
            X (np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.pipeline.predict(X)

    def get_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Squared Error.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: RMSE score.
        """
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)


def run_training_and_submission(load_cached_data=True):
    """
    Orchestrates the full pipeline: feature extraction, training, validation,
    and submission generation.

    Args:
        load_cached_data (bool): If True, attempts to load features from disk cache.
    """
    # 1. Feature Extraction
    # Uses the DualBackboneExtractor (Swin Large + ConvNeXt Large)
    extractor = DualBackboneExtractor()

    # Load features for all splits
    # train/val targets are floats, test targets are IDs (strings)
    train_feats, train_meta, train_targets = extractor.get_features(
        "train", load_cached_data
    )
    val_feats, val_meta, val_targets = extractor.get_features("val", load_cached_data)
    test_feats, test_meta, test_ids = extractor.get_features("test", load_cached_data)

    # 2. Feature Fusion
    # Concatenate Image Embeddings (approx 3072 dim) with Metadata (12 dim)
    X_train = np.hstack([train_feats, train_meta])
    X_val = np.hstack([val_feats, val_meta])
    X_test = np.hstack([test_feats, test_meta])

    # 3. Model Training
    model = RidgeRegressor()
    model.fit(X_train, train_targets)

    # 4. Validation
    val_preds = model.predict(X_val)
    val_rmse = model.get_rmse(val_targets, val_preds)
    print(f"Validation RMSE: {val_rmse}")

    # 5. Submission Generation
    test_preds = model.predict(X_test)

    submission_df = pd.DataFrame({"Id": test_ids, "Pawpularity": test_preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
