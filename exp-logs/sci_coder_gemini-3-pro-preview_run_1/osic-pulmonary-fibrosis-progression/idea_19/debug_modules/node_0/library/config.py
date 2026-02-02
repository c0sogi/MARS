import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Tabular-Gated Dual-View Network pipeline.
    Handles paths, hyperparameters, and reproducibility settings.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Idea-specific directory for caching processed data
    IDEA_NAME = "idea_19"
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Checkpoints and Submissions
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Input Data Paths
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output File Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"{IDEA_NAME}_best_model.pth")

    # ==========================================
    # Data Preprocessing & Augmentation
    # ==========================================
    # Image Resolution (Native EfficientNet-B0)
    IMG_SIZE = 224

    # Tri-Slab Generation
    NUM_SLABS = 3
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs
    VIEWS = ["axial", "coronal"]

    # Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Tabular Features
    # 'Percent' is critical for the clinical gating mechanism
    NUMERICAL_COLS = ["Age", "Percent"]
    CATEGORICAL_COLS = ["Sex", "SmokingStatus"]

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # Dimensions
    # EfficientNet-B0 outputs 1280 dim features
    BACKBONE_DIM = 1280
    # We project tabular features UP to match visual fidelity
    TABULAR_EMBED_DIM = 1280

    # Attention Fusion
    ATTN_HEADS = 4
    ATTN_LAYERS = 1
    DROPOUT = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 8

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Metric & Loss Constants
    # ==========================================
    # Modified Laplace Log Likelihood constraints
    METRIC_CLIP_ERR = 1000.0
    METRIC_MIN_CONF = 70.0

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized: {cls.CACHE_DIR}, {cls.CHECKPOINT_DIR}")

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")
