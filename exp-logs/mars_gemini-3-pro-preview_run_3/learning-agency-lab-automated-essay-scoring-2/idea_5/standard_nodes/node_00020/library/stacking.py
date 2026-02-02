import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import get_logger


class LGBMStacker:
    """
    Implements the second-stage Stacking model using LightGBM.
    This model takes embeddings (from the backbone) and explicit meta-features
    as input to predict the final essay score.
    """

    def __init__(self, config: Config):
        """
        Initialize the stacker with configuration parameters.

        Args:
            config (Config): Configuration object containing model parameters and paths.
        """
        self.config = config
        self.logger = get_logger("stacking.log")
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        """
        Trains the LightGBM regressor on the provided features and targets.
        Performs an internal train/validation split to enable early stopping unless eval_set is provided.

        Args:
            X (np.ndarray): Feature matrix (concatenation of embeddings and meta-features).
            y (np.ndarray): Target scores (integers or floats).
            eval_set (tuple): Optional (X_val, y_val) tuple for explicit validation.
        """
        self.logger.info(f"Starting LightGBM training with input shape: {X.shape}")

        if eval_set is None:
            # Create an internal validation split (20%) for early stopping
            # We use the fixed seed from config to ensure reproducibility
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.config.seed
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = eval_set

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Prepare parameters
        params = self.config.lgbm_params.copy()

        # Callbacks for monitoring and early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100),
        ]

        # Train the model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get("n_estimators", 2000),
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log the best score achieved
        if self.model.best_score:
            for dataset_name, metrics in self.model.best_score.items():
                for metric_name, score in metrics.items():
                    # Printing full precision as requested
                    self.logger.info(f"Best {dataset_name} {metric_name}: {score}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the given features.
        Applies clipping to [1, 6] and rounding to nearest integer.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted integer scores.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call train() or load() first."
            )

        # Predict continuous scores
        # uses best_iteration automatically if early stopping was used
        preds_continuous = self.model.predict(
            X, num_iteration=self.model.best_iteration
        )

        # Post-processing: Clip to valid range [1, 6]
        preds_clipped = np.clip(preds_continuous, 1, 6)

        # Round to nearest integer
        preds_rounded = np.round(preds_clipped).astype(int)

        return preds_rounded

    def save(self, filename: str = "lgbm_model.txt"):
        """
        Saves the trained LightGBM model to the configured output directory.

        Args:
            filename (str): Name of the model file.
        """
        if self.model is None:
            self.logger.warning("No model to save.")
            return

        output_path = os.path.join(self.config.output_dir, filename)
        self.model.save_model(output_path)
        self.logger.info(f"LightGBM model saved to {output_path}")

    def load(self, filename: str = "lgbm_model.txt"):
        """
        Loads a LightGBM model from the configured output directory.

        Args:
            filename (str): Name of the model file.
        """
        model_path = os.path.join(self.config.output_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = lgb.Booster(model_file=model_path)
        self.logger.info(f"LightGBM model loaded from {model_path}")

    def make_submission(
        self, essay_ids: list, X_test: np.ndarray, filename: str = "submission.csv"
    ):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            essay_ids (list): List of essay IDs corresponding to the test features.
            X_test (np.ndarray): Test feature matrix.
            filename (str): Name of the submission file.

        Returns:
            pd.DataFrame: The submission DataFrame.
        """
        self.logger.info(f"Generating submission for {len(essay_ids)} essays...")

        # Generate predictions
        predictions = self.predict(X_test)

        # Create DataFrame
        submission_df = pd.DataFrame({"essay_id": essay_ids, "score": predictions})

        # Ensure submission directory exists
        os.makedirs(self.config.submission_dir, exist_ok=True)
        output_path = os.path.join(self.config.submission_dir, filename)

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        self.logger.info(f"Submission saved to {output_path}")

        return submission_df
