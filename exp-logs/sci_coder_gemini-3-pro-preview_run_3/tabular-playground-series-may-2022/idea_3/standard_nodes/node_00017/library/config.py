import os
import torch


class Config:
    """
    Configuration class for the Two-Stage Denoising Autoencoder (DAE) Pipeline.
    Handles file paths, hyperparameters, model architecture, and feature definitions.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs for data loading, bounded by a reasonable number
    NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 4)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Input Directories (Metadata based on the provided task description)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Input (for sample submission structure)
    INPUT_DIR = "./input"
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_3"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Paths
    PRETRAINED_MODEL_PATH = os.path.join(WORKING_DIR, "dae_autoencoder.pth")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_classifier.pth")

    # -------------------------------------------------------------------------
    # Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
    LOAD_CACHED_DATA = True

    # Column Definitions
    # Continuous features: f_00 to f_26, f_28 (f_27 is string)
    CONTINUOUS_COLS = [f"f_{i:02d}" for i in range(29) if i != 27]

    # Categorical/Discrete features to be embedded
    # f_29 and f_30 are discrete indicators
    DISCRETE_COLS = ["f_29", "f_30"]

    # String feature to be decomposed into characters
    STRING_COL = "f_27"
    F27_SEQ_LEN = 10  # Length of the string in f_27

    # Engineered feature name
    COUNT_COL = "unique_char_count"

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Encoder Backbone (Funnel MLP)
    # Reducing width: Input -> 512 -> 256 -> 128
    ENCODER_LAYERS = [512, 256, 128]

    # Decoder (Mirroring Encoder for Pretraining)
    # Expanding width: 128 -> 256 -> 512 -> Input
    DECODER_LAYERS = [128, 256, 512]

    # Embedding Dimensions
    # Characters (A-Z, etc.) and small discrete features
    CHAR_EMBEDDING_DIM = 8
    DISCRETE_EMBEDDING_DIM = 4

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 2048  # Large batch size for stable DAE training

    # Stage 1: Unsupervised Pretraining (Removed per Lesson 13)
    # Direct Supervised Training is preferred.

    # Stage 2: Supervised Training
    FINETUNE_EPOCHS = 30
    FINETUNE_MAX_LR = 1e-2  # Max LR for OneCycle policy

    # Optimization
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5  # Early stopping patience

    @staticmethod
    def setup():
        """
        Creates necessary output directories for the pipeline.
        Should be called at the start of the execution.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
