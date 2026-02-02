import os
import torch


class Config:
    # ==========================================
    # 1. PATH CONFIGURATION
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 12)
    WORK_DIR = "./working/idea_12"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    MODEL_DIR = WORK_DIR  # Where to save model checkpoints

    # Cache file for processed tensors
    CACHE_FILE = os.path.join(CACHE_DIR, "processed_data.npz")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Source files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # 2. DATA CONFIGURATION
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    CHANNELS = 3  # Band 1, Band 2, Average

    # Augmentation
    ROTATION_ANGLES = [0, 90, 180, 270]

    # Normalization
    # Note: Min/Max scaling is calculated per-channel dynamically or using fixed stats if preferred.
    # The strategy specifies Independent Per-Channel Min-Max Scaling.

    # ==========================================
    # 3. MODEL ARCHITECTURE
    # ==========================================
    # Sustained-Width Backbone: Stage 1 -> Stage 2 -> Stage 3 -> Stage 4
    BACKBONE_FILTERS = [64, 128, 128, 128]

    # Aggregation Head
    # Path A (Local-Spatial) Output Dim: 128 * 4 * 4 = 2048
    # Path B (Global-Peak) Output Dim: 128

    # Metadata Branch
    META_HIDDEN_DIM = 32  # Dimension for processing inc_angle

    # Classification Head
    DROPOUT_RATE = 0.2
    NUM_CLASSES = 1  # Binary classification (sigmoid output)

    # ==========================================
    # 4. TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 2e-4  # "Low and Slow"
    WEIGHT_DECAY = (
        1e-4  # Only if needed, strictly speaking logic says "if val loss > train loss"
    )
    # but setting a small default is standard practice.

    MAX_EPOCHS = 100
    PATIENCE = 15  # Early stopping patience

    # Scheduler
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup on import to ensure directories exist
Config.setup()
