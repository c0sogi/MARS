import os
import torch
import numpy as np
import random
import pandas as pd


class Config:
    # --- Hardware & Compute ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Supplemental data
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Configuration ---
    INPUT_SIZE = 1024  # 1024x1024 resolution
    NUM_CLASSES = 4782  # Total unique characters in unicode_translation.csv

    # --- Training Hyperparameters ---
    BATCH_SIZE = 4
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP = 10.0
    SEED = 42

    # --- Model Architecture ---
    MODEL_NAME = "hrnet_w32"

    # --- Inference Configuration ---
    CONF_THRESHOLD = 0.1
    MAX_PREDS = 1200

    @staticmethod
    def seed_everything(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_class_mappings(cls):
        """
        Loads the unicode translation file to create mappings between
        Unicode character strings and integer class IDs.

        Returns:
            char2id (dict): Mapping from Unicode string (e.g., 'U+003F') to int ID.
            id2char (dict): Mapping from int ID to Unicode string.
        """
        if not os.path.exists(cls.UNICODE_MAP_PATH):
            raise FileNotFoundError(f"Unicode map not found at {cls.UNICODE_MAP_PATH}")

        df = pd.read_csv(cls.UNICODE_MAP_PATH)

        # The file is expected to have a 'Unicode' column
        if "Unicode" not in df.columns:
            raise ValueError("unicode_translation.csv must contain a 'Unicode' column.")

        unicodes = df["Unicode"].values

        char2id = {code: idx for idx, code in enumerate(unicodes)}
        id2char = {idx: code for idx, code in enumerate(unicodes)}

        return char2id, id2char
