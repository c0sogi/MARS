import os
import torch


class Config:
    """
    Configuration for the Anchored Symmetric-Attention Dual-Axis Network (ASA-DAN).
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific Idea Directory (Cache & Outputs)
    IDEA_NAME = "idea_34"
    IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
    CACHE_DIR = IDEA_DIR  # Cache files go here
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")
    LOG_DIR = os.path.join(IDEA_DIR, "logs")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Submission Output
    SUBMISSION_PATH = os.path.join(IDEA_DIR, "submission.csv")

    # ==========================================
    # 2. Data Preprocessing & Augmentation
    # ==========================================
    # Image Generation (Tri-Slab)
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Number of slabs per view
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Views
    VIEWS = ["axial", "coronal"]

    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Dataloader
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # Dimensionality
    # We maintain native backbone dimensionality to preserve texture signal
    FEATURE_DIM = 1280

    # Tabular Features
    # Raw features used for the parametric head
    TAB_COLS = ["Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
    # Features used for the Gated Linear Unit (GLU) expansion
    CONTEXT_COLS = ["Age", "Sex", "SmokingStatus", "Percent"]

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 32  # A100 allows larger batch size
    EPOCHS = 50
    LR = 1e-4  # Learning rate
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # Cycle length
    ETA_MIN = 1e-6  # Minimum LR

    # Early Stopping
    PATIENCE = 8  # Strict patience as requested

    # Loss Clipping
    MAX_ERROR = 1000  # Threshold for metric calculation
    CONFIDENCE_CLIP = 70  # Minimum confidence value

    # ==========================================
    # 5. Debugging & Reproducibility
    # ==========================================
    DEBUG = False  # Set to True for quick checks
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the experiment.
        """
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

        # Set reproducibility
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """
        Sets seeds for reproducibility.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
