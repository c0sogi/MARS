import os
import torch
import random
import numpy as np


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    # Using 'idea_2' as the working directory for this specific iteration
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Checkpoint and Output files
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 4

    # --- Data Parameters ---
    PATCH_SIZE = 128
    BATCH_SIZE = 32

    # Debugging / Development flags
    # Set to True and a small number to test pipeline quickly
    DEBUG = False
    DEBUG_SAMPLES = 50

    # --- Model Architecture (ResUNet) ---
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    # Encoder/Decoder feature depths
    FEATURES = [64, 128, 256, 512]

    # --- Training Hyperparameters ---
    # Extended duration for convergence on augmented data
    NUM_EPOCHS = 350
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # --- Scheduler (Cosine Annealing) ---
    # Decoupled horizon: T_MAX significantly larger than NUM_EPOCHS
    # to keep learning rate active throughout training
    T_MAX = 1000
    ETA_MIN = 1e-6

    # --- Inference ---
    # Test-Time Augmentation enabled
    TTA_ENABLED = True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for outputs.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def seed_everything(cls, seed=None):
        """
        Sets random seeds for reproducibility.
        """
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            True  # True for speed, False for exact reproducibility
        )


# Automatically setup directories when config is imported
Config.setup()
