import os
import torch


class Config:
    """
    Global configuration for the Explicit Additive Dual-Stream Network (EADS-Net) experiment.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    IDEA_NAME = "idea_15"
    DEBUG = False  # Set to True to run on a small subset for testing

    # -------------------------------------------------------------------------
    # Directory & File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{IDEA_NAME}"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Image preprocessing
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices
    SLICE_AREA_THRESHOLD = 0.5  # Threshold for selecting boundary slices

    # Tabular preprocessing
    TIME_SCALE = 0.01  # Scaling factor for relative weeks (t_rel)

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b2"
    PRETRAINED = True
    PROJECTION_DIM = 128  # Dimension for projecting image features
    DROPOUT_RATE = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Number of dataloader workers

    # Optimizer settings (Differential Learning Rates)
    LEARNING_RATE_BACKBONE = 1e-4  # Lower LR for fine-tuning the backbone
    LEARNING_RATE_HEAD = 1e-3  # Higher LR for the new stream heads
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    # Note: We use a higher patience because uncertainty estimation often
    # takes longer to calibrate than the mean prediction.
    PATIENCE = 15

    # -------------------------------------------------------------------------
    # Metric & Loss Constants
    # -------------------------------------------------------------------------
    SIGMA_CLIP = 70.0  # Minimum confidence value for metric calculation
    MAX_ERROR = 1000.0  # Maximum error cap for metric calculation
    # Increased from 0.05 to 0.1 to prevent numerical instability in loss
    # 0.1 scaled units approx 80ml, which is safe relative to SIGMA_CLIP (70ml)
    EPSILON = 0.1  # Numerical stability floor for uncertainty

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print(f"--- Configuration ({Config.IDEA_NAME}) ---")
        print(f"Device: {Config.DEVICE}")
        print(f"Backbone: {Config.BACKBONE_NAME}")
        print(f"Epochs: {Config.EPOCHS}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"LR Backbone: {Config.LEARNING_RATE_BACKBONE}")
        print(f"LR Head: {Config.LEARNING_RATE_HEAD}")
        print(f"Cache Dir: {Config.CACHE_DIR}")
        print("---------------------------------------")
