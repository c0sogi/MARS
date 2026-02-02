import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Denoising project.
    """

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for idea_1 specific artifacts (checkpoints, processed data)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(CACHE_DIR, "best_model.pth")

    # --- Data Hyperparameters ---
    # Training on full images is inefficient given the small dataset size (92 images).
    # We extract random patches of this size during training.
    PATCH_SIZE = 128

    # Images are loaded in grayscale
    NUM_CHANNELS = 1

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 16
    NUM_EPOCHS = 400
    LEARNING_RATE = 1e-3

    # Scheduler Horizon (Decoupled from Epochs)
    # Cite solution_lesson_node_00007: Horizon Decoupling
    T_MAX = 1500

    # Early Stopping
    PATIENCE = 25

    # --- Hardware Settings ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using a moderate number of workers for data loading
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration setup complete. Device: {cls.DEVICE}")
