import os
import random
import numpy as np
import torch
import pandas as pd


class Config:
    """
    Global configuration for the Depth-Conditioned LinkNet project.
    Handles hyperparameters, paths, seeding, and dynamic calculation of normalization stats.
    """

    def __init__(self, **kwargs):
        # ==========================
        # Paths
        # ==========================
        self.METADATA_DIR = "./metadata"
        self.TRAIN_CSV = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_CSV = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_CSV = os.path.join(self.METADATA_DIR, "test.csv")

        self.INPUT_ROOT = "./input"
        self.IMAGE_DIR_TRAIN = os.path.join(self.INPUT_ROOT, "train", "images")
        self.MASK_DIR_TRAIN = os.path.join(self.INPUT_ROOT, "train", "masks")
        self.IMAGE_DIR_TEST = os.path.join(self.INPUT_ROOT, "test", "images")

        self.WORKING_DIR = "./working/idea_1"
        self.CHECKPOINT_PATH = os.path.join(self.WORKING_DIR, "best_model.pth")

        self.SUBMISSION_DIR = "./submission"
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # Create necessary directories
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # ==========================
        # Data Hyperparameters
        # ==========================
        self.ORIG_SHAPE = (101, 101)
        self.INPUT_SHAPE = (128, 128)  # Padded size for U-Net/LinkNet divisibility
        self.CHANNELS = 1  # Grayscale
        self.NUM_CLASSES = 1  # Binary segmentation

        # ==========================
        # Training Hyperparameters
        # ==========================
        self.SEED = 42
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.NUM_WORKERS = 2  # CPU workers for dataloader
        self.PATIENCE = 10  # Early stopping patience
        self.THRESHOLD = 0.5  # Probability threshold for mask generation

        # ==========================
        # Compute Settings
        # ==========================
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # ==========================
        # Overrides
        # ==========================
        for k, v in kwargs.items():
            setattr(self, k, v)

        # ==========================
        # Setup
        # ==========================
        self.set_seed()

        # Calculate Depth Statistics for Normalization
        # We compute this dynamically from the training metadata to ensure accuracy
        self.DEPTH_MEAN, self.DEPTH_STD = self._get_depth_stats()

    def set_seed(self):
        """Sets the random seed for reproducibility."""
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(self.SEED)

    def _get_depth_stats(self):
        """
        Calculates mean and std of the depth ('z') feature from the training set.
        Used for standardizing the depth input in the model.
        """
        if os.path.exists(self.TRAIN_CSV):
            try:
                df = pd.read_csv(self.TRAIN_CSV)
                mean_z = df["z"].mean()
                std_z = df["z"].std()
                return mean_z, std_z
            except Exception as e:
                print(
                    f"Warning: Could not calculate depth stats from {self.TRAIN_CSV}. Error: {e}"
                )
                return 0.0, 1.0
        else:
            print(
                f"Warning: Metadata file {self.TRAIN_CSV} not found. Using default depth stats."
            )
            return 0.0, 1.0

    def __repr__(self):
        return str(self.__dict__)
