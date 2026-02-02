import os
import torch


class Config:
    """
    Global configuration for Salt Segmentation Task.
    Implements settings for:
    - High-Capacity Deep Residual U-Net with Coordinate Attention
    - Consistent Compound Loss (BCE + Dice + Lovasz)
    - Cyclic Cosine Annealing Schedule
    - Snapshot Ensembling
    """

    # -------------------------------------------------------------------------
    # 1. Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # 2. File Systems & Paths
    # -------------------------------------------------------------------------
    # Read-only Input
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Artifacts (Idea 11)
    WORKING_DIR = "./working/idea_11"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Final Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 3. Data Parameters
    # -------------------------------------------------------------------------
    ORIG_SIZE = 101
    INPUT_SIZE = 128  # Padded size (Reflection Padding) for U-Net divisibility
    CHANNELS = 1  # Seismic images are grayscale
    DEPTH_CHANNELS = 1  # Depth 'z' feature map

    # Data Loading
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Augmentation Strategy
    AUGMENTATION_FLIP_ONLY = True  # Restrict to Horizontal Flips only

    # -------------------------------------------------------------------------
    # 4. Model & Optimization Hyperparameters
    # -------------------------------------------------------------------------
    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Cyclic Scheduler
    NUM_CYCLES = 3
    EPOCHS_PER_CYCLE = 50
    TOTAL_EPOCHS = NUM_CYCLES * EPOCHS_PER_CYCLE  # 150 Total

    # Consistent Compound Loss Weights
    # L = L_BCE + L_Dice_Sample + 0.05 * L_Lovasz
    WEIGHT_BCE = 1.0
    WEIGHT_DICE = 1.0
    WEIGHT_LOVASZ = 0.05

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # 5. Debugging & Control
    # -------------------------------------------------------------------------
    DEBUG = False
    DEBUG_DATA_LIMIT = None  # None implies full dataset

    @classmethod
    def setup(cls):
        """
        Initialize the directory structure for the experiment.
        Must be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def set_debug_mode(cls, debug=True, epochs_per_cycle=2, data_limit=100):
        """
        Enable or disable debug mode to speed up development iterations.

        Args:
            debug (bool): Whether to enable debug mode.
            epochs_per_cycle (int): Number of epochs per cycle in debug mode.
            data_limit (int): Maximum number of samples to load.
        """
        cls.DEBUG = debug
        if debug:
            cls.EPOCHS_PER_CYCLE = epochs_per_cycle
            cls.TOTAL_EPOCHS = cls.NUM_CYCLES * cls.EPOCHS_PER_CYCLE
            cls.DEBUG_DATA_LIMIT = data_limit
            print(
                f"[Config] Debug mode ENABLED. Epochs: {cls.TOTAL_EPOCHS}, Data Limit: {cls.DEBUG_DATA_LIMIT}"
            )
        else:
            cls.EPOCHS_PER_CYCLE = 50
            cls.TOTAL_EPOCHS = 150
            cls.DEBUG_DATA_LIMIT = None
            print("[Config] Debug mode DISABLED. Using full training schedule.")
