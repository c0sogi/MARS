import os
import torch


class Config:
    """
    Configuration class for Contrail Identification.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # --- Project Identification ---
    PROJECT_NAME = "contrail_segmentation"
    IDEA_NAME = "idea_2"
    SEED = 42

    # --- Data Paths ---
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # --- Working & Output Paths ---
    # Working directory for intermediate files (checkpoints, cache)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Parameters ---
    IMAGE_SIZE = 256
    N_CHANNELS = 6  # Ash composite (3) + Temporal Difference (3) = 6 channels
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3

    # --- Model Architecture ---
    ENCODER_NAME = "efficientnet_b0"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = None  # Logits returned by model

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Stabilization & Ensembling ---
    TOP_K_CHECKPOINTS = 5  # Number of best checkpoints to save for averaging

    # --- Debugging & Development ---
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 300  # Subset size when DEBUG is True

    @classmethod
    def setup_directories(cls):
        """
        Creates the necessary directory structure for the project.
        This ensures that checkpoint, cache, and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when the module is imported
Config.setup_directories()
