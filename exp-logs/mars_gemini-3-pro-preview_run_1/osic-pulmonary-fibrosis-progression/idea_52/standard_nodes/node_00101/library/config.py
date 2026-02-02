import os
import torch


class Config:
    """
    Centralized configuration for the Normalized Shared-Latent Holistic Network (NSL-HN).
    """

    # ==========================================
    # 1. General Settings
    # ==========================================
    PROJECT_NAME = "NSL-HN_Lung_Decline"
    IDEA_ID = "idea_52"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SIZE = 50  # Number of samples to use in debug mode

    # ==========================================
    # 2. File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts
    WORKING_DIR = os.path.join("./working", IDEA_ID)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 3. Data Processing (Tri-Slab & DICOM)
    # ==========================================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLABS = 3  # Number of slabs per view (Axial/Coronal)
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Lung Window Settings (Hounsfield Units)
    WINDOW_LEVEL = -650
    WINDOW_WIDTH = 1500

    # ==========================================
    # 4. Model Architecture (NSL-HN)
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_PRETRAINED = True

    # Dimensions
    BACKBONE_OUT_DIM = 1280  # Native output dim of EfficientNet-B0 (no projection)
    LATENT_DIM = 128  # Dimension of the Shared Latent Vector (T_lat)
    FUSED_DIM = 1280  # Dimension after fusion (matches backbone)

    # Regularization
    DROPOUT_RATE = 0.1

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16  # Adjusted for 2x Backbones + Attention on A100
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # For AdamW
    PATIENCE = 8  # Strict early stopping

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 6. Metric Constants
    # ==========================================
    # Modified Laplace Log Likelihood constants
    MAX_ERROR = 1000.0  # Error threshold in ml
    SIGMA_CLIP = 70.0  # Minimum confidence clipping in ml

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"\n{'='*20} CONFIGURATION {'='*20}")
        print(f"Project: {cls.PROJECT_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Backbone: {cls.BACKBONE_NAME}")
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"{'='*55}\n")
