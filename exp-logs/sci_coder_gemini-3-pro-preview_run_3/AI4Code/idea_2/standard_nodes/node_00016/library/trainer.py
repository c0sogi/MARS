import os
import lightgbm as lgb
from library.config import Config
from library.utils import set_seed
from library.dataset import TabularDataset


class ModelTrainer:
    """
    Manages the training of the LightGBM Regressor.
    """

    def __init__(self, debug_limit: int = None):
        set_seed(Config.SEED)
        self.save_path = Config.MODEL_SAVE_PATH

        # Load Data
        self.train_data = TabularDataset("train")
        self.val_data = TabularDataset("val")

        self.model = None

    def train(self):
        print("Loading training data...")
        X_train, y_train, _ = self.train_data.load()
        print("Loading validation data...")
        X_val, y_val, _ = self.val_data.load()

        print(
            f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples."
        )

        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        print("Starting LightGBM training...")
        self.model = lgb.train(
            Config.LGBM_PARAMS,
            train_set,
            num_boost_round=Config.NUM_BOOST_ROUND,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=50),
            ],
        )

        print(f"Saving model to {self.save_path}...")
        self.model.save_model(self.save_path)
        print("Training complete.")
