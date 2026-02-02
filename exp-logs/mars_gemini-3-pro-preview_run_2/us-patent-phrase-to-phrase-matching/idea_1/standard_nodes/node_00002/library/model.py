import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from library.config import Config
from library.utils import compute_pearson_correlation


class SimilarityRegressor:
    """
    Implements the Linear Regression Head on top of frozen embeddings.
    Uses Ridge Regression to predict similarity scores based on
    concatenated embeddings and their absolute difference.
    """

    def __init__(self):
        """
        Initializes the Ridge regressor with parameters from Config.
        """
        self.alpha = Config.RIDGE_ALPHA
        self.random_state = Config.SEED
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)

    def _construct_features(self, anchors, targets):
        """
        Constructs the feature vector for the linear model.
        Features: Concatenation of [anchor_embedding, target_embedding, |anchor - target|].

        Args:
            anchors (np.ndarray): Anchor embeddings of shape (N, D).
            targets (np.ndarray): Target embeddings of shape (N, D).

        Returns:
            np.ndarray: Combined features of shape (N, 3*D).
        """
        # Calculate element-wise absolute difference
        diff = np.abs(anchors - targets)

        # Concatenate u, v, and |u-v|
        features = np.concatenate([anchors, targets, diff], axis=1)
        return features

    def fit(
        self,
        train_anchors,
        train_targets,
        train_scores,
        val_anchors=None,
        val_targets=None,
        val_scores=None,
    ):
        """
        Trains the Ridge regression model.

        Args:
            train_anchors (np.ndarray): Training anchor embeddings.
            train_targets (np.ndarray): Training target embeddings.
            train_scores (np.ndarray): Training target scores.
            val_anchors (np.ndarray, optional): Validation anchor embeddings.
            val_targets (np.ndarray, optional): Validation target embeddings.
            val_scores (np.ndarray, optional): Validation target scores.
        """
        # Construct training features
        X_train = self._construct_features(train_anchors, train_targets)

        # Train the model
        self.model.fit(X_train, train_scores)

        # Evaluate on Training Data
        train_preds = self.predict(train_anchors, train_targets)
        train_corr = compute_pearson_correlation(train_scores, train_preds)
        print(f"Training Pearson Correlation: {train_corr}")

        # Evaluate on Validation Data (if provided)
        if (
            val_anchors is not None
            and val_targets is not None
            and val_scores is not None
        ):
            val_preds = self.predict(val_anchors, val_targets)
            val_corr = compute_pearson_correlation(val_scores, val_preds)
            print(f"Validation Pearson Correlation: {val_corr}")

    def predict(self, anchors, targets):
        """
        Generates similarity score predictions.

        Args:
            anchors (np.ndarray): Anchor embeddings.
            targets (np.ndarray): Target embeddings.

        Returns:
            np.ndarray: Predicted scores clipped to [0, 1].
        """
        X = self._construct_features(anchors, targets)
        preds = self.model.predict(X)

        # Clip predictions to valid range [0, 1]
        return np.clip(preds, 0.0, 1.0)


def generate_submission(model, test_df, test_anchors, test_targets):
    """
    Generates predictions for the test set and saves them to the submission file.

    Args:
        model (SimilarityRegressor): The trained model instance.
        test_df (pd.DataFrame): The test dataframe containing 'id'.
        test_anchors (np.ndarray): Test anchor embeddings.
        test_targets (np.ndarray): Test target embeddings.
    """
    # Generate predictions
    preds = model.predict(test_anchors, test_targets)

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_df["id"], "score": preds})

    # Ensure output directory exists (handled by Config setup, but good practice)
    # Config.SUBMISSION_FILE includes the directory path

    # Save to CSV
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
