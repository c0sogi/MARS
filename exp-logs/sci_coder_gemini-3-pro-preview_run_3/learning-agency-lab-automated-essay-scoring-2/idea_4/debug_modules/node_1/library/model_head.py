import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import get_logger

logger = get_logger("ModelHead")


class StackingTrainer:
    """
    Manages the training and inference of the LightGBM stacking head.
    """

    def __init__(self):
        """
        Initialize the trainer with parameters from Config.
        """
        self.params = Config.get_lgbm_params()
        self.model = None

    def _prepare_features(self, embeddings, meta_features):
        """
        Concatenates embeddings and meta-features into a single feature matrix.

        Args:
            embeddings (np.ndarray): Backbone embeddings (N, D).
            meta_features (np.ndarray): Meta features (N, K).

        Returns:
            np.ndarray: Combined features (N, D+K).
        """
        if meta_features is None:
            return embeddings

        # Ensure meta_features is 2D
        if len(meta_features.shape) == 1:
            meta_features = meta_features.reshape(-1, 1)

        # Check length consistency
        if embeddings.shape[0] != meta_features.shape[0]:
            raise ValueError(
                f"Shape mismatch: Embeddings {embeddings.shape}, Meta {meta_features.shape}"
            )

        return np.hstack([embeddings, meta_features])

    def fit(
        self,
        embeddings,
        meta_features,
        labels,
        val_embeddings=None,
        val_meta_features=None,
        val_labels=None,
    ):
        """
        Trains the LightGBM model.

        Args:
            embeddings: Training embeddings.
            meta_features: Training meta-features.
            labels: Training target scores.
            val_embeddings: Optional validation embeddings.
            val_meta_features: Optional validation meta-features.
            val_labels: Optional validation target scores.
        """
        logger.info("Preparing data for LightGBM training...")
        X = self._prepare_features(embeddings, meta_features)
        y = labels

        valid_sets = []
        valid_names = []

        # Determine validation strategy
        if val_embeddings is not None and val_labels is not None:
            logger.info("Using provided validation set.")
            X_train = X
            y_train = y
            X_val = self._prepare_features(val_embeddings, val_meta_features)
            y_val = val_labels

            train_ds = lgb.Dataset(X_train, label=y_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

            valid_sets = [train_ds, val_ds]
            valid_names = ["train", "valid"]
        else:
            # Internal split if no validation set provided (crucial for early stopping)
            logger.info(
                "No validation set provided. Performing internal 90/10 split for Early Stopping."
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=Config.seed, shuffle=True
            )

            train_ds = lgb.Dataset(X_train, label=y_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

            valid_sets = [train_ds, val_ds]
            valid_names = ["train", "valid"]

        # Extract num_boost_round from params (n_estimators)
        # lgb.train uses 'num_boost_round', sklearn API uses 'n_estimators'
        num_boost_round = self.params.pop("n_estimators", 5000)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)

        # Configure callbacks
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        ]

        logger.info(f"Training LightGBM (Boost rounds: {num_boost_round})...")
        self.model = lgb.train(
            self.params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Log best score
        if self.model.best_score:
            best_iter = self.model.best_iteration
            for data_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    # Printing full precision as requested
                    logger.info(
                        f"Best {data_name} {metric_name} at iteration {best_iter}: {score}"
                    )

    def predict(self, embeddings, meta_features):
        """
        Generates predictions using the trained model.

        Args:
            embeddings: Input embeddings.
            meta_features: Input meta-features.

        Returns:
            np.ndarray: Predicted scores.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        X = self._prepare_features(embeddings, meta_features)
        # Predict using the best iteration found during training
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, output_dir, filename="lgbm_model.txt"):
        """
        Saves the trained booster to a file.
        """
        if self.model is None:
            logger.warning("No model to save.")
            return

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        self.model.save_model(path)
        logger.info(f"LightGBM model saved to {path}")

    def load(self, model_path):
        """
        Loads a booster from a file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = lgb.Booster(model_file=model_path)
        logger.info(f"LightGBM model loaded from {model_path}")


def make_submission(ids, predictions, output_path):
    """
    Formats predictions and saves the submission CSV.

    Args:
        ids (array-like): Essay IDs.
        predictions (array-like): Predicted scores.
        output_path (str): Path to save the CSV.
    """
    # Ensure predictions are numpy array
    preds = np.array(predictions)

    # Post-processing for QWK: Clip to [1, 6] and round to nearest integer
    preds_processed = np.clip(preds, 1, 6).round().astype(int)

    df = pd.DataFrame({"essay_id": ids, "score": preds_processed})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Submission file saved to {output_path} with shape {df.shape}")
