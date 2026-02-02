import os
import torch


class Config:
    """
    Configuration class for the Balanced-Bottleneck Shared-Latent Network (BBSL-Net).
    Centralizes all hyperparameters, file paths, and environment settings.
    """

    # ==========================================
    # 1. Environment & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    # Input Data Roots
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Generated previously)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Specific to Idea 56)
    WORKING_DIR = "./working/idea_56"

    # Caching Directory for processed images/features
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model Checkpoints
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Processing Parameters
    # ==========================================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    IN_CHANS = 3  # RGB (MIPs mapped to channels)

    # Tri-Slab Generation Logic
    N_SLICES = 3  # Number of slabs per view
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Debugging / Development
    DEBUG = False  # Set to True to run on a subset
    DEBUG_SAMPLE_SIZE = 20  # Number of samples if DEBUG is True

    # ==========================================
    # 4. Model Architecture (BBSL-Net)
    # ==========================================
    BACKBONE_NAME = "tf_efficientnet_b0_ns"

    # Dimensionality
    LATENT_DIM = 128  # Shared latent dimension (T_lat)
    VISUAL_DIM = 1280  # Native backbone output dimension (EfficientNet-B0)
    FUSED_DIM = 1280  # Dimension after attention fusion

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimizer
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # Cycle length
    ETA_MIN = 1e-6  # Minimum learning rate

    # Early Stopping
    PATIENCE = 8  # Strict patience as requested

    # ==========================================
    # 6. Metric & Loss Constants
    # ==========================================
    # Modified Laplace Log Likelihood constants
    MAX_ERROR = 1000.0  # Clip absolute error at 1000 ml
    MIN_CONFIDENCE = 70.0  # Clip confidence at 70 ml

    @classmethod
    def setup(cls):
        """
        Ensures that all necessary working and output directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_debug_config(cls):
        """
        Returns a dictionary or modified state for debugging purposes.
        """
        return {
            "debug": cls.DEBUG,
            "sample_size": cls.DEBUG_SAMPLE_SIZE,
            "epochs": 2 if cls.DEBUG else cls.EPOCHS,
        }


# Automatically setup directories when imported
Config.setup()
