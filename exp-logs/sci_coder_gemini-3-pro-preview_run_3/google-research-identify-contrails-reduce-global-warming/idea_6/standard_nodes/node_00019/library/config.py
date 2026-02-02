import os
import torch


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 6)
    WORKING_DIR = "./working/idea_6"

    # Sub-directories for artifacts
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- System & Reproducibility ---
    SEED = 42
    NUM_WORKERS = 8  # 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    IMG_SIZE = 256
    # 6 Channels: Ash (Band 11, 13, 14, 15) + Temporal Difference (Ash_t - Ash_t-1)
    N_CHANNELS = 6

    # --- Model ---
    # Scaled up backbone as per strategy
    BACKBONE = "convnext_small"
    ENCODER_WEIGHTS = "imagenet"

    # --- Training Hyperparameters ---
    EPOCHS = 40
    # A100 40GB can handle larger batches, but convnext_small is heavier.
    # We use 32 and can rely on gradient accumulation in the training loop if needed.
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler
    T_MAX = 40  # Matches EPOCHS for Cosine Annealing
    ETA_MIN = 1e-6

    # Optimization & Logging
    TOP_K_CHECKPOINTS = 5  # Keep best 5 for averaging
    VALIDATION_INTERVAL = 1

    # --- Inference ---
    USE_TTA = True  # Test Time Augmentation enabled
    THRESHOLD = 0.5

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
