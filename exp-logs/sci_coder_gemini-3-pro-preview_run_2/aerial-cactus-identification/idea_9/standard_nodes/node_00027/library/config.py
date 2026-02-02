import os


class Config:
    """
    Configuration for the Custom Deeply Supervised Narrow ResNet-UNet experiment.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data & Preprocessing
    # ==========================================
    IMAGE_SIZE = 32
    NUM_CLASSES = 1

    # Debugging: Set to a small integer (e.g., 100) to train on a subset
    DEBUG_SUBSET_SIZE = None

    # ==========================================
    # Model Architecture
    # ==========================================
    # Narrow ResNet Encoder channels as specified in the idea
    ENCODER_CHANNELS = [16, 32, 64]

    # Deep Supervision Settings
    USE_DEEP_SUPERVISION = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    BATCH_SIZE = 128
    MAX_EPOCHS = 30

    # Optimizer (AdamW) & Scheduler (Cosine Annealing)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 7

    # Loss Function Weights
    # L_total = 0.5 * L_semantic + 0.5 * L_detail
    LOSS_WEIGHT_SEMANTIC = 0.5
    LOSS_WEIGHT_DETAIL = 0.5

    # ==========================================
    # Augmentation
    # ==========================================
    # Only flips allowed, no rotations/color jitter
    AUG_HFLIP_PROB = 0.5
    AUG_VFLIP_PROB = 0.5

    # ==========================================
    # Hardware / Runtime
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_cache_path(cls, filename):
        """
        Helper to get full path for a file in the working directory.
        """
        return os.path.join(cls.WORKING_DIR, filename)
