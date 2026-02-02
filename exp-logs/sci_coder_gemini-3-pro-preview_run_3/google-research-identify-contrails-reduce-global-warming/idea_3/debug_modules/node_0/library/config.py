import os
import torch


class Config:
    """
    Central configuration for the Contrail Identification pipeline.
    Defines hyperparameters, file paths, and model specifications.
    """

    # --- General Configuration ---
    SEED = 42
    PROJECT_NAME = "contrails-segmentation-idea-3"

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for Idea 3 (U-Net++ EfficientNet-B4)
    WORKING_DIR = "./working/idea_3"

    # Sub-directories for artifacts
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Metadata Paths ---
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # --- Data Parameters ---
    IMAGE_SIZE = 256
    # 6 Channels: 3 for Ash Color (t) + 3 for Ash Color (t) - Ash Color (t-1)
    INPUT_CHANNELS = 6
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # --- Model Architecture ---
    # Using U-Net++ with EfficientNet-B4 backbone as per strategy
    BACKBONE = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32  # Fits comfortably in A100 40GB with B4 backbone
    EPOCHS = 35
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler (Cosine Annealing)
    T_MAX = 35
    ETA_MIN = 1e-6

    # --- Checkpointing & Averaging Strategy ---
    # Online Top-K Checkpointing
    CHECKPOINT_TOP_K = 5

    # Convergence-Aware Weight Averaging:
    # Only average checkpoints saved after 50% of training is complete.
    # 50% of 35 epochs is 17.5, so we start averaging after epoch 17.
    AVERAGE_START_EPOCH = 17

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Debugging / Development ---
    # Toggle this to True to run on a small subset of data for testing pipeline
    DEBUG = False

    @classmethod
    def get_dataset_limit(cls):
        """
        Returns the maximum number of samples to load.
        Useful for debugging to avoid loading the full dataset.
        """
        if cls.DEBUG:
            return 500  # Small subset for quick debugging
        return None  # None implies loading the full dataset
