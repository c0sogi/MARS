import os
import torch


class Config:
    """
    Global configuration for the Hybrid Transformer-Funnel Network pipeline.
    Defines file paths, hyperparameters, and model architecture specifications.
    """

    # --------------------------------------------------------------------------
    # Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (using stratified metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic data processing)
    CACHE_PROCESSED_DATA = os.path.join(CACHE_DIR, "processed_data.pt")
    CACHE_VOCAB = os.path.join(CACHE_DIR, "vocab_info.npy")
    CACHE_SCALER = os.path.join(CACHE_DIR, "scaler.npy")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    ID_COL = "id"
    TARGET_COL = "target"

    # Feature Definitions
    # f_27 is decomposed; f_29 and f_30 are treated as categorical
    # Continuous features: f_00..f_28 (excluding f_27) + engineered 'unique_character_count'
    CONTINUOUS_FEATURE_NAMES = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]

    # Categorical Configuration
    # 10 characters from f_27 + f_29 + f_30 = 12 tokens in the sequence
    CATEGORICAL_SEQ_LEN = 12

    # --------------------------------------------------------------------------
    # Model Architecture (Hybrid Transformer-Funnel)
    # --------------------------------------------------------------------------
    # Categorical Branch
    EMBED_DIM = 32  # Capacity for high-cardinality signal
    TRANSFORMER_HEADS = 4  # Multi-head attention for token interactions
    TRANSFORMER_LAYERS = 1  # Single layer to learn dependencies without overfitting
    TRANSFORMER_FF_DIM = 128  # Feed-forward dimension inside Transformer

    # Funnel MLP Backbone
    # Decreasing width for feature compression
    FUNNEL_LAYERS = [512, 256, 128]
    DROPOUT = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024  # Moderate batch size for regularization
    EPOCHS = 20  # Sufficient for convergence with Early Stopping
    LEARNING_RATE = 1e-3  # Max LR for OneCycle Policy
    WEIGHT_DECAY = 1e-5  # Calibrated regularization (avoiding default 1e-2)
    PATIENCE = 5  # Early stopping patience

    # --------------------------------------------------------------------------
    # Compute & Debugging
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DEBUG = False  # Set to True to use a subset of data
    DEBUG_SAMPLES = 10000  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Ensures all working and output directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon import
Config.setup()
