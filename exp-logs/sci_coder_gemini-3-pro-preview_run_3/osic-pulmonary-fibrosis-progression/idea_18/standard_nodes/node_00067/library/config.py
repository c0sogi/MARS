import os


class Config:
    """
    Configuration for the Cascaded Latent-Residual Network (CLR-Net) pipeline.
    Centralizes file paths, hyperparameters, and normalization constants.
    """

    # -------------------------------------------------------------------------
    # File System Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories for Idea 18
    IDEA_DIR = "./working/idea_18"
    CACHE_DIR = os.path.join(IDEA_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing & Normalization
    # -------------------------------------------------------------------------
    SEED = 42
    N_SLICES = 3
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2

    # Relative Time Scaling (Weeks * 0.01)
    TIME_SCALE = 0.01

    # Normalization Statistics (Derived from Data Analysis)
    # Used for Z-score standardization of inputs and target

    # Target Variable (FVC)
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Clinical Features
    PERCENT_MEAN = 76.9105
    PERCENT_STD = 19.1970
    AGE_MEAN = 67.5825
    AGE_STD = 6.6259

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "tf_efficientnet_b2_ns"
    LATENT_DIM = 64
    DROPOUT = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LR_BACKBONE = 1e-4  # Lower LR for fine-tuning feature extractor
    LR_HEAD = 1e-3  # Higher LR for MLPs
    WEIGHT_DECAY = 1e-2
    T_MAX = 50  # Cosine Annealing duration

    # -------------------------------------------------------------------------
    # Inference / Metric
    # -------------------------------------------------------------------------
    CONFIDENCE_CLIP = 70.0  # Minimum confidence value (sigma)
    MAX_ERROR_CLIP = 1000.0  # Metric error threshold

    @staticmethod
    def setup():
        """
        Creates the necessary working directories for the pipeline.
        Should be called at the start of execution.
        """
        os.makedirs(Config.IDEA_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
