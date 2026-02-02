import os
import numpy as np
import lightgbm as lgb
from library.config import Config
from library.utils import set_seed
from library.dataset import NotebookSequenceDataset
from library.model import LGBMModel


class ModelTrainer:
    """
    Manages the training and validation lifecycle of the LightGBM model.
    """

    def __init__(self, debug_limit: int = None):
        set_seed(Config.SEED)
        self.save_path = Config.MODEL_SAVE_PATH
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        print("Initializing ModelTrainer for LightGBM...")

        # Load Data
        self.train_dataset = NotebookSequenceDataset(
            split="train", load_cached_data=True, debug_limit=debug_limit
        )
        self.val_dataset = NotebookSequenceDataset(
            split="val", load_cached_data=True, debug_limit=debug_limit
        )

        # Prepare Data for LightGBM
        # We extract the features and labels from the dataset samples list
        print("Preparing LightGBM datasets...")
        self.X_train, self.y_train = self._prepare_data(self.train_dataset)
        self.X_val, self.y_val = self._prepare_data(self.val_dataset)

        self.model = LGBMModel()

    def _prepare_data(self, dataset):
        features = []
        targets = []
        for sample in dataset.samples:
            features.append(sample["features"])
            targets.append(sample["target"])
        return np.array(features), np.array(targets)

    def train(self):
        print("Starting LightGBM training...")

        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)

        self.model.train(train_data, val_data)
        self.model.save(self.save_path)

        print(f"Training complete. Model saved to {self.save_path}")
