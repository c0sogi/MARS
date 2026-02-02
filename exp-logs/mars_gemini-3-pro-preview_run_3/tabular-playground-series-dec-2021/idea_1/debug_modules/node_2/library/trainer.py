import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_loader import ForestDataLoader
from library.model import GBDTWrapper


class Trainer:
    """
    Orchestrates the training and evaluation pipeline for Cover Type prediction.
    """

    def __init__(self):
        """
        Initialize the Trainer with a data loader.
        """
        self.loader = ForestDataLoader()
        self.model = None

    def train(self, load_cached_data=True, **model_params):
        """
        Loads data and trains the GBDT model.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed data.
            **model_params: Hyperparameters to override in the model configuration
                            (e.g., n_estimators, learning_rate).
        """
        print("Loading training and validation data...")
        X_train, y_train = self.loader.get_data(
            "train", load_cached_data=load_cached_data
        )
        X_val, y_val = self.loader.get_data("val", load_cached_data=load_cached_data)

        # Initialize the model with provided parameters
        self.model = GBDTWrapper(**model_params)

        # Fit the model
        # The wrapper handles early stopping and metric printing
        self.model.fit(X_train, y_train, X_val, y_val)

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set and creates a submission file.

        Args:
            load_cached_data (bool): Whether to use cached preprocessed test data.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        print("Loading test data...")
        X_test, _ = self.loader.get_data("test", load_cached_data=load_cached_data)

        # Retrieve Test IDs
        # The data loader removes IDs from X_test, so we must load them from the raw file.
        # We must replicate the loader's slicing logic if DEBUG is enabled to ensure alignment.
        df_test_raw = pd.read_parquet(Config.TEST_DATA_PATH)

        if Config.DEBUG:
            df_test_raw = df_test_raw.iloc[: Config.DEBUG_SAMPLE_SIZE]

        test_ids = df_test_raw[Config.ID_COL]

        # Generate submission file
        self.model.generate_submission(X_test, test_ids)


def run_training(
    debug=False, load_cached_data=True, n_estimators=2000, learning_rate=0.1, **kwargs
):
    """
    Main entry point to run the training pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data.
        load_cached_data (bool): If True, attempts to load processed data from cache.
        n_estimators (int): Maximum number of boosting rounds.
        learning_rate (float): Learning rate for the booster.
        **kwargs: Additional model hyperparameters.
    """
    # Set random seed for reproducibility
    np.random.seed(Config.SEED)

    # Configure global debug state
    Config.DEBUG = debug
    if debug:
        print(f"Running in DEBUG mode with sample size {Config.DEBUG_SAMPLE_SIZE}")

    # Prepare model parameters
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    # Update with any additional arguments
    model_params.update(kwargs)

    # Initialize and run trainer
    trainer = Trainer()

    # Train
    trainer.train(load_cached_data=load_cached_data, **model_params)

    # Predict and Submit
    trainer.predict(load_cached_data=load_cached_data)
