import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("model_trainer")


class GradientBoostingTrainer:
    """
    Wrapper class for LightGBM training and inference using the low-level API.
    """

    def __init__(self):
        """
        Initializes the trainer with parameters from Config.
        """
        self.model = None
        self.params = Config.MODEL_PARAMS.copy()

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the LightGBM model with early stopping and validation monitoring.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training targets (0-indexed integers).
            X_val (pd.DataFrame): Validation features.
            y_val (np.ndarray): Validation targets (0-indexed integers).
        """
        logger.info("Preparing LightGBM datasets...")

        # Create LightGBM datasets
        # Reference ensures proper bin alignment between train and val
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Define callbacks for logging and early stopping
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        logger.info(f"Starting training with {Config.NUM_BOOST_ROUND} max rounds...")
        logger.info(f"Model parameters: {self.params}")

        try:
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                num_boost_round=Config.NUM_BOOST_ROUND,
                callbacks=callbacks,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e

        # Save the trained model
        self._save_model()

    def _save_model(self):
        """
        Saves the trained model to the configured path.
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            self.model.save_model(Config.MODEL_SAVE_PATH)
            logger.info(f"Model saved to {Config.MODEL_SAVE_PATH}")
        else:
            logger.warning("No model to save.")

    def predict(self, X_test):
        """
        Generates class predictions for the test set.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predicted class indices (0-indexed).
        """
        # Load model if not currently in memory (e.g., if running inference separately)
        if self.model is None:
            if os.path.exists(Config.MODEL_SAVE_PATH):
                logger.info(f"Loading model from {Config.MODEL_SAVE_PATH}...")
                self.model = lgb.Booster(model_file=Config.MODEL_SAVE_PATH)
            else:
                raise RuntimeError(
                    "Model has not been trained and no saved artifact found at destination."
                )

        logger.info(f"Predicting on {len(X_test)} samples...")

        # LightGBM predict returns raw probabilities for multiclass (N_samples, N_classes)
        y_pred_prob = self.model.predict(X_test)

        # Convert probabilities to class indices using argmax
        y_pred_indices = np.argmax(y_pred_prob, axis=1)

        return y_pred_indices
