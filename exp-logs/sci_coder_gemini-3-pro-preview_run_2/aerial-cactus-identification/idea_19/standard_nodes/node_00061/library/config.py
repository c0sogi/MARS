import os
import torch


class Config:
    """
    Centralized configuration for Idea 19: Custom Wide ResNet with Dense Residual Projections.
    """

    # --- Project Identification ---
    IDEA_NAME = "idea_19"

    # --- Directories ---
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories (Write Allowed)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # --- File Paths ---
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Parameters ---
    IMAGE_SIZE = 32
    CHANNELS = 3
    NUM_CLASSES = 1

    # --- Model Architecture ---
    # Wide Channel Configuration as per "Optimized Baseline" strategy
    # Backbone stages will use these channel widths
    MODEL_CHANNELS = [32, 64, 128]

    # --- Training Hyperparameters ---
    # Homogeneous Ensemble: 5 independent seeds
    SEEDS = [0, 1, 2, 3, 4]

    # Training Loop
    NUM_EPOCHS = 15
    BATCH_SIZE = 128

    # Optimization (AdamW + Cosine Annealing)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Regularization
    EARLY_STOPPING_PATIENCE = 5

    # --- Debugging / Development ---
    # If set to an integer (e.g., 100), only this many samples will be used for training/validation
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_model_path(cls, seed):
        """Returns the file path for saving/loading a model checkpoint for a specific seed."""
        return os.path.join(cls.WORKING_DIR, f"model_seed_{seed}.pth")


# Automatically setup directories when imported
Config.setup()
