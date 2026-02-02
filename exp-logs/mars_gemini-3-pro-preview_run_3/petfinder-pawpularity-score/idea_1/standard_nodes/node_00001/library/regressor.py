import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import compute_rmse, save_submission, set_seed


class RidgeRegressor:
    """
    A wrapper around sklearn's Ridge regression that handles feature concatenation,
    scaling, and prediction clipping for the Pawpularity score.
    """

    def __init__(self, alpha=10.0, random_state=42):
        """
        Args:
            alpha (float): Regularization strength for Ridge regression.
            random_state (int): Seed for reproducibility.
        """
        self.alpha = alpha
        self.random_state = random_state
        # Pipeline: Scale features -> Ridge Regression
        self.model = make_pipeline(
            StandardScaler(), Ridge(alpha=self.alpha, random_state=self.random_state)
        )

    def _prepare_features(self, img_features, meta_features):
        """
        Concatenates image embeddings and metadata features.

        Args:
            img_features (np.array): Shape (N, 1280)
            meta_features (np.array): Shape (N, 12)

        Returns:
            np.array: Combined features of shape (N, 1292)
        """
        # Ensure inputs are numpy arrays
        if not isinstance(img_features, np.ndarray):
            img_features = np.array(img_features)
        if not isinstance(meta_features, np.ndarray):
            meta_features = np.array(meta_features)

        return np.hstack([img_features, meta_features])

    def fit(self, img_features, meta_features, targets):
        """
        Fits the Ridge model to the training data.

        Args:
            img_features (np.array): Training image features.
            meta_features (np.array): Training metadata features.
            targets (np.array): Training target values.
        """
        X = self._prepare_features(img_features, meta_features)
        self.model.fit(X, targets)

        # Calculate and print training RMSE
        train_preds = self.model.predict(X)
        train_preds = np.clip(train_preds, 1.0, 100.0)
        train_rmse = compute_rmse(targets, train_preds)
        print(f"Training RMSE: {train_rmse}")

    def predict(self, img_features, meta_features):
        """
        Generates predictions for the given features.

        Args:
            img_features (np.array): Image features.
            meta_features (np.array): Metadata features.

        Returns:
            np.array: Predicted Pawpularity scores clipped to [1, 100].
        """
        X = self._prepare_features(img_features, meta_features)
        preds = self.model.predict(X)
        # Clip predictions to the valid range of the dataset
        return np.clip(preds, 1.0, 100.0)


def train_and_evaluate(data):
    """
    Orchestrates the training, validation, and submission generation process.

    Args:
        data (dict): Dictionary containing 'train', 'val', and 'test' tuples
                     as returned by feature_extractor.extract_features().
    """
    set_seed(Config.SEED)

    # 1. Unpack Data
    print("Unpacking data...")
    train_img, train_meta, train_y = data["train"]
    val_img, val_meta, val_y = data["val"]
    test_img, test_meta, test_ids = data["test"]

    # 2. Initialize Model
    # Using hyperparameters from Config
    print(f"Initializing Ridge Regressor with alpha={Config.RIDGE_ALPHA}...")
    regressor = RidgeRegressor(alpha=Config.RIDGE_ALPHA, random_state=Config.SEED)

    # 3. Train
    print("Starting training...")
    regressor.fit(train_img, train_meta, train_y)

    # 4. Validate
    print("Running validation...")
    val_preds = regressor.predict(val_img, val_meta)
    val_rmse = compute_rmse(val_y, val_preds)
    # Print full precision as requested
    print(f"Validation RMSE: {val_rmse}")

    # 5. Generate Submission
    print("Generating predictions for test set...")
    test_preds = regressor.predict(test_img, test_meta)

    save_submission(test_ids, test_preds, Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
