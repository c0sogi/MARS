import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Prior-Guided Attention-Readout Network (PGAR-Net).
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Idea-specific directory (Idea 68: PGAR-Net)
    IDEA_ID = "idea_68"
    OUTPUT_DIR = os.path.join(WORKING_DIR, IDEA_ID)
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # DICOM Directories (Relative to INPUT_ROOT as per metadata)
    # Note: Metadata contains relative paths, but we define roots here for convenience
    TRAIN_DICOM_ROOT = INPUT_ROOT
    TEST_DICOM_ROOT = INPUT_ROOT

    # ==========================================
    # 2. Hyperparameters
    # ==========================================
    SEED = 42
    DEBUG = False

    # Training
    EPOCHS = 50
    BATCH_SIZE = 16  # Adjusted for VRAM and EfficientNet-B0
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict early stopping
    NUM_WORKERS = 4

    # Debugging / Development
    # If set to an integer, limits the number of samples for rapid iteration
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    # ==========================================
    # 3. Data Processing & Augmentation
    # ==========================================
    # Image Specs
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Tri-slab architecture
    SLAB_OVERLAP = 0.15
    IN_CHANNELS = 3  # RGB (MIP mapped to channels or repeated)

    # Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 4. Model Architecture (PGAR-Net)
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_PRETRAINED = True

    # Dimensions
    VISUAL_FEATURE_DIM = 1280  # Native B0 output (no compression)
    TABULAR_LATENT_DIM = 128  # Shared latent vector size
    CONTEXT_HIDDEN_DIM = 64  # Projection size for interaction streams

    # Tabular Features
    TABULAR_COLS = ["Age", "Sex", "SmokingStatus", "Percent"]
    NUMERICAL_COLS = ["Age", "Percent"]
    CATEGORICAL_COLS = ["Sex", "SmokingStatus"]

    # ==========================================
    # 5. Metric & Loss
    # ==========================================
    # Constants for the modified Laplace Log Likelihood
    MAX_ERROR_CLIP = 1000.0
    MIN_CONFIDENCE_CLIP = 70.0

    @classmethod
    def setup(cls):
        """
        Sets up the environment: creates directories and sets random seeds.
        """
        # Create necessary directories
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        cls.seed_everything(cls.SEED)

        # Device configuration
        cls.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Config: Setup complete. Using device: {cls.DEVICE}")
        print(f"Config: Output directory set to {cls.OUTPUT_DIR}")

    @staticmethod
    def seed_everything(seed):
        """
        Seeds all random number generators for reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
