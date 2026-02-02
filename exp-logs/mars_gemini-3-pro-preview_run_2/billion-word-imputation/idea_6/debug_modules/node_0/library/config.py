import os
import torch


class Config:
    """
    Configuration for the Probabilistic Beam-Search Cascade solution.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory for Artifacts
    WORKING_DIR = "./working/idea_6"

    # Cache Paths (Processed Datasets)
    # Using Parquet for efficient storage and loading
    LOCATOR_TRAIN_CACHE = os.path.join(WORKING_DIR, "locator_train.parquet")
    LOCATOR_VAL_CACHE = os.path.join(WORKING_DIR, "locator_val.parquet")
    INFILLER_TRAIN_CACHE = os.path.join(WORKING_DIR, "infiller_train.parquet")
    INFILLER_VAL_CACHE = os.path.join(WORKING_DIR, "infiller_val.parquet")

    # Model Checkpoints
    LOCATOR_MODEL_PATH = os.path.join(WORKING_DIR, "best_locator.pth")
    INFILLER_MODEL_PATH = os.path.join(WORKING_DIR, "best_infiller.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing
    # --------------------------------------------------------------------------
    # Maximum sequence length for tokenization
    MAX_LEN = 128

    # Dataset Sampling
    # Scaled to 1M samples to ensure robust generalization while fitting in 24h
    TRAIN_SAMPLE_SIZE = 1_000_000
    VAL_SAMPLE_SIZE = 50_000

    # --------------------------------------------------------------------------
    # Stage 1: Syntactic Locator (DeBERTa-v3)
    # --------------------------------------------------------------------------
    # DeBERTa-v3 uses disentangled attention, superior for relative positioning
    LOCATOR_MODEL_NAME = "microsoft/deberta-v3-base"
    LOCATOR_BATCH_SIZE = 32
    LOCATOR_LR = 2e-5
    LOCATOR_EPOCHS = 3
    LOCATOR_LABEL_SMOOTHING = 0.1  # Helps prevent overconfidence for beam search

    # --------------------------------------------------------------------------
    # Stage 2: Semantic In-Filler (RoBERTa-Large)
    # --------------------------------------------------------------------------
    # RoBERTa-Large provides high-capacity semantic knowledge
    INFILLER_MODEL_NAME = "roberta-large"
    INFILLER_BATCH_SIZE = 16  # Smaller batch size due to larger model
    INFILLER_LR = 1e-5  # Lower LR to preserve pre-trained knowledge
    INFILLER_EPOCHS = 3

    # --------------------------------------------------------------------------
    # Inference Strategy
    # --------------------------------------------------------------------------
    BEAM_K = 3  # Number of candidates to generate from Locator for In-Filler

    def __init__(self):
        """
        Initialize configuration and ensure working directories exist.
        """
        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
