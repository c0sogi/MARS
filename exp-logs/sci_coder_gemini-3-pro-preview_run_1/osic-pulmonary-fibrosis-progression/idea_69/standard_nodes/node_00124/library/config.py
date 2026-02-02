import os
import torch
import numpy as np
import random


class Config:
    # ==========================
    # File Paths & Directories
    # ==========================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific experiment directory for caching and checkpoints
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_69")
    CACHE_DIR = os.path.join(IDEA_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================
    # Data Preprocessing
    # ==========================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLABS = 3  # Tri-slab design (Top, Middle, Bottom)
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Normalization (ImageNet stats)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # Tabular Features
    # Note: 'Weeks' is handled dynamically as a relative value in the model logic
    # The Prior Encoder strictly uses demographic/clinical features.
    PRIOR_NUM_COLS = ["Percent", "Age"]
    PRIOR_CAT_COLS = ["Sex", "SmokingStatus"]

    # ==========================
    # Model Architecture (PGA-Net)
    # ==========================
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_DIM = 1280  # EfficientNet-B0 output dim (without projection)

    # Latent Dimensions
    SHARED_LATENT_DIM = 128  # T_lat: Output of the tabular encoder
    ALIGN_DIM = 1280  # T_align: Projected tabular dim to match visual backbone
    COMPRESS_DIM = (
        64  # Dimension for Visual and Interaction streams before final assembly
    )
    ASSEMBLED_DIM = 256  # Final vector dim (128 Prior + 64 Visual + 64 Interaction)

    # Head constraints
    MAX_FVC_ERROR = 1000.0
    MIN_CONFIDENCE = 70.0

    # ==========================
    # Training Hyperparameters
    # ==========================
    SEED = 42
    BATCH_SIZE = 16  # Adjusted for memory constraints with dual backbones
    N_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict patience for early stopping
    NUM_WORKERS = 4

    # Debug / Development
    DEBUG = False  # Set to True to run on a subset of data
    DEBUG_SIZE = 50  # Number of samples if DEBUG is True

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
