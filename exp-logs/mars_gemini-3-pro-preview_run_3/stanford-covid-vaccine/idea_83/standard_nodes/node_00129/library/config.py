import os
import torch


class Config:
    """
    Configuration for the RNA Degradation Prediction experiment.
    Implements the settings for the Deep Hierarchical BiGRU with Deep Supervision strategy.
    """

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_83"
    SUBMISSION_DIR = "./submission"

    # Input Files (Parquet Metadata)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone: Deep Hierarchical BiGRU
    HIDDEN_DIM = 384  # Per direction (Total 768)
    NUM_LAYERS = 4  # Deep 4-layer backbone
    DROPOUT = 0.1  # Conservative regularization

    # Input Specification
    SEQ_LEN = 107
    INPUT_CHANNELS = 14  # 4 (Seq) + 3 (Struct) + 7 (Loop)

    # Output Specification
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    SEQ_SCORED = 68  # Only first 68 positions are scored
    SCORED_COLUMNS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Gradient clipping for stability

    # Deep Supervision
    DEEP_SUPERVISION_WEIGHT = 0.3

    # Optimization
    PATIENCE = 5  # Early stopping patience

    # ==========================================
    # Hardware & Computation
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2

    # ==========================================
    # Debugging & Development
    # ==========================================
    DEBUG = False  # Set to True to run on a subset
    DEBUG_SIZE = 100  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_target_columns(cls):
        """
        Returns the list of all target columns in the dataset.
        """
        return ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    @classmethod
    def get_scored_columns(cls):
        """
        Returns the list of columns used for the competition metric.
        """
        return cls.SCORED_COLUMNS
