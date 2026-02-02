import os
import torch


class Config:
    """
    Configuration for the Balanced-Bottleneck Shared-Latent Network (BBSL-Net).
    """

    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    # Root directory for input data (Read-Only)
    INPUT_DIR = "./input"

    # Metadata directory (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this specific idea (Write Allowed)
    # Used for caching processed data and saving models
    IDEA_ID = "idea_58"
    WORKING_DIR = os.path.join("./working", IDEA_ID)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # 2. Data Preprocessing
    # ==========================================
    # Image Configuration
    # Native resolution for EfficientNet-B0 to avoid overfitting
    IMG_SIZE = 224

    # Tri-Slab Configuration
    # We use 3 slabs per view (Axial/Coronal)
    SLAB_COUNT = 3

    # Normalization (ImageNet Standards)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Tabular Features
    NUMERICAL_COLS = ["Weeks", "Percent", "Age"]
    CATEGORICAL_COLS = ["Sex", "SmokingStatus"]

    # ==========================================
    # 3. Model Architecture (BBSL-Net)
    # ==========================================
    # Visual Backbone
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_DIM = 1280  # Native output dim of B0 GAP
    PRETRAINED = True

    # Shared Latent Topology
    LATENT_DIM = 128  # Dimension for T_lat and H_compressed

    # Dropout rates
    DROPOUT_RATE = 0.1

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 50

    # Early Stopping
    PATIENCE = 8  # Strict patience as per design

    # Metric Constraints
    SIGMA_MIN = 70.0
    ERROR_MAX = 1000.0

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, debug=False):
        """
        Initialize configuration with optional debug mode.

        Args:
            debug (bool): If True, reduces epochs and dataset size for quick testing.
        """
        self.debug = debug

        if self.debug:
            self.EPOCHS = 2
            self.BATCH_SIZE = 8
            self.NUM_WORKERS = 0  # Avoid multiprocessing overhead in debug

        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.SUBMISSION_PATH), exist_ok=True)

    def __str__(self):
        """Print configuration for logging."""
        return (
            f"Config(ID={self.IDEA_ID}, "
            f"Backbone={self.BACKBONE_NAME}, "
            f"ImgSize={self.IMG_SIZE}, "
            f"LatentDim={self.LATENT_DIM}, "
            f"Debug={self.debug})"
        )
