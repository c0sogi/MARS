import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Centralizes hyperparameters, file paths, and environment setup.
    """

    # ==============================
    # Data Dimensions & Targets
    # ==============================
    SEQ_LEN = 107
    PRED_LEN = 68

    # The columns used for scoring and training (subset of ground truth)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==============================
    # Model Architecture
    # ==============================
    # Embedding breakdown
    EMBED_DIM_SEQ = 128
    EMBED_DIM_LOOP = 64
    EMBED_DIM_DIST = 64
    EMBED_DIM = EMBED_DIM_SEQ + EMBED_DIM_LOOP + EMBED_DIM_DIST  # Total: 256

    HIDDEN_DIM = 384
    NUM_LAYERS = 6
    DROPOUT = 0.2

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    DEEP_SUPERVISION_WEIGHT = 0.5

    # ==============================
    # Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_65"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (using generated metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Reproducibility
    # ==============================
    SEED = 42

    @classmethod
    def setup_environment(cls, seed=None):
        """
        Prepares the environment for the experiment.
        1. Creates necessary output directories.
        2. Sets random seeds for reproducibility.
        3. Returns the compute device.

        Args:
            seed (int, optional): Random seed. Defaults to Config.SEED.

        Returns:
            torch.device: The device (CPU or CUDA) available.
        """
        # 1. Create Directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # 2. Set Random Seeds
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 3. Return Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
