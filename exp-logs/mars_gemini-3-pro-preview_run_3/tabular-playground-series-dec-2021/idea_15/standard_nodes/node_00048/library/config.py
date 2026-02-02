import os
import torch


class Config:
    """
    Global configuration for the Parallel Factorized DCN-ResNet experiment.
    """

    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # ==========================================
    # Data Paths
    # ==========================================
    # Metadata Parquet Files (Pre-split 80/20 Stratified)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Cache Paths (for deterministic processing)
    # ==========================================
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Architecture: Parallel Factorized DCN-ResNet
    MODEL_NAME = "ParallelFactorizedDCNResNet"

    # Branch 1: Factorized DCN (Low-Rank Decomposition)
    DCN_RANK = 16  # Rank (r) for W = U * V^T

    # Branch 2: Wide ResNet Backbone
    HIDDEN_DIM = 512  # Width of hidden layers
    RESNET_BLOCKS = 2  # Number of residual blocks in the backbone

    # General Model Params
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience

    # Scheduler: Cosine Annealing
    SCHEDULER_T_MAX = 60  # Matches EPOCHS
    SCHEDULER_ETA_MIN = 0.0

    # ==========================================
    # Hardware & Runtime
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
