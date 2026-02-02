import os
import torch


class Config:
    """
    Configuration module for the Deeply-Supervised (Annealed) Dual-View
    Asymmetric Parallel Vector-DCN-ResNet task.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data and models
    WORKING_DIR = "./working/idea_43"

    # Data Paths (using generated metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # 2. Data Configuration
    # ==========================================
    SEED = 42
    NUM_CLASSES = 7  # Cover_Type classes 1-7

    # Feature Engineering Flags
    USE_QUANTILE_TRANSFORM = True  # For Statistical View
    USE_ASPECT_CYCLICAL = True  # Sin/Cos Aspect
    USE_HYDRO_DISTANCE = True  # Euclidean distance
    USE_HYDRO_ELEVATION = True  # Absolute elevation
    USE_AMENITIES_MEAN = True  # Global context

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    # Branch 1: Asymmetric Vector-Based DCN (Warm-Start)
    DCN_LAYERS = 3
    DCN_RANK = 1  # Vector-based (Rank-1)
    DCN_INIT_STD = 1e-4  # Near-Zero initialization

    # Branch 2: Deeply-Supervised ResNet Backbone
    BACKBONE_BLOCKS = 5
    HIDDEN_DIM = 256
    DROPOUT = 0.2
    # Attach Auxiliary Head after Block 3 (Index 2 in 0-based list)
    AUX_BLOCK_INDEX = 2

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Optimization Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1  # Aggressive decay
    SCHEDULER_PATIENCE = 5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Loss Annealing (Lambda decay for Aux head)
    ANNEAL_START = 0.3
    ANNEAL_END = 0.0

    # ==========================================
    # 5. System & Hardware
    # ==========================================
    NUM_WORKERS = 12
    # Detect device automatically
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Disable strict determinism for performance (Lesson 00070)
    DETERMINISTIC_CUDNN = False

    @classmethod
    def setup(cls):
        """
        Create necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
