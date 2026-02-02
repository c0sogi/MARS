import os
import torch


class Config:
    """
    Configuration for Contrail Identification Pipeline.
    Strategy: U-Net with ConvNeXt-Small on Multi-Order Temporal Composites.
    """

    # --- General Experiment Settings ---
    EXPERIMENT_NAME = "idea_9"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # --- Paths ---
    ROOT_DIR = "./input"
    METADATA_DIR = "./metadata"
    OUTPUT_DIR = f"./working/{EXPERIMENT_NAME}"
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
    SUBMISSION_FILE = "./submission/submission.csv"

    # --- Data Configuration ---
    IMAGE_SIZE = 256
    INPUT_CHANNELS = 9  # 3 (Current Ash) + 3 (Velocity Diff) + 3 (Acceleration Diff)
    NUM_WORKERS = 12

    # Ash Composite Bands (GOES-16 ABI Band Numbers)
    # Band 11: 8.4 µm, Band 14: 11.2 µm, Band 15: 12.3 µm
    ASH_BANDS = [11, 14, 15]

    # Temporal Sequence Indices (0-indexed)
    # The dataset provides sequences with n_times_before=4.
    # Index 4 is the labeled frame (t).
    # Index 3 is t-1.
    # Index 2 is t-2.
    TEMPORAL_INDICES = [4, 3, 2]

    # --- Model Architecture ---
    BACKBONE = "convnext_small"
    PRETRAINED = True

    # --- Training Hyperparameters ---
    EPOCHS = 40
    BATCH_SIZE = 32  # Optimized for A100 40GB
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000

    # --- Optimization & Scheduling ---
    # Cosine Annealing Scheduler
    T_MAX = 40
    MIN_LR = 1e-6

    # --- Loss Function ---
    # Hybrid Dice + BCE
    SMOOTH = 1e-6

    # --- Checkpointing & Validation ---
    SAVE_TOP_K = 5  # Save top 5 best models based on Dice score
    VAL_CHECK_INTERVAL = 1.0  # Check validation every epoch

    # --- Inference ---
    # Test Time Augmentation (TTA) enabled in inference pipeline
    USE_TTA = True

    # --- Hardware ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories for the experiment.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTION_DIR, exist_ok=True)

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(cls.SUBMISSION_FILE), exist_ok=True)


# Automatically create directories upon import
Config.setup()
