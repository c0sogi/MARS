import os
import torch


class Config:
    """
    Configuration class for the Metric-Aligned Cascaded Anchor Network (MACAN).
    Contains hyperparameters, file paths, and normalization constants.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use in debug mode
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    DICOM_DIR = (
        "./input"  # Root directory for DICOMs (metadata contains relative paths)
    )

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Write Allowed)
    # Using specific idea folder as requested
    WORKING_DIR = "./working/idea_28"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Preprocessing & Augmentation
    # ==========================================
    # Image Parameters
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # 1 Anchor (Max Area) + 2 Boundaries

    # Radiographic Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Normalization Constants (Derived from EDA)
    # Used for Z-scoring the target variable FVC
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Relative Time Scaling
    TIME_SCALE = 0.01  # Scale weeks to be closer to network active range

    # ==========================================
    # Model Architecture (MACAN)
    # ==========================================
    BACKBONE_NAME = "efficientnet_b2"
    # Dimensions
    PROJECTION_DIM = 64  # Dimension of image features projection
    HIDDEN_DIM = 128  # Dimension of clinical anchor MLP hidden layer
    LATENT_DIM = 64  # Shared latent dimension for cascade

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimization
    # Differential Learning Rates
    LR_BACKBONE = 1e-4  # Lower LR for fine-tuning pre-trained weights
    LR_HEAD = 1e-3  # Higher LR for the MLP streams and head
    WEIGHT_DECAY = 1e-2  # Standard AdamW weight decay

    # Scheduler (Cosine Annealing)
    T_MAX = 50  # Should match EPOCHS for full cycle
    ETA_MIN = 1e-6

    # ==========================================
    # Metric & Inference
    # ==========================================
    # Constants for Metric-Aligned Laplace Log Likelihood
    SQRT_2 = 1.41421356

    # Post-processing constraints
    CONFIDENCE_CLIP = 70.0  # Minimum uncertainty (ml)
    MAX_ERROR_CLIP = 1000.0  # Max error penalty (ml)

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        Should be called at the start of the execution.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")
