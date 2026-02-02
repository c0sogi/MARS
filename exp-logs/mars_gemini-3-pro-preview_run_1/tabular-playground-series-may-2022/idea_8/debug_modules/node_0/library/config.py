import os
import torch


class Config:
    """
    Configuration for the Denoising Granular Unified Transformer (DeGUT) pipeline.
    Acts as a central source of truth for hyperparameters, paths, and settings.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging/testing

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for the DeGUT model artifacts
    WORKING_DIR = "./working/degut_model"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata file paths (Pre-split stratified data)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Model Architecture (DeGUT)
    # -------------------------------------------------------------------------
    # Dimensions for the granular tokens
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    ACTIVATION = "gelu"

    # Sequence length buffer: 30 numerical features + ~10 char tokens + 1 CLS
    MAX_SEQ_LEN = 64

    # -------------------------------------------------------------------------
    # Denoising & Optimization Objectives
    # -------------------------------------------------------------------------
    # Probability of masking a token (numerical or sequence) for reconstruction
    MASK_PROB = 0.15

    # Weight for the reconstruction loss term in the composite loss function
    # Loss = BCE + LOSS_LAMBDA * (MSE_num + CE_seq)
    LOSS_LAMBDA = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # A100 GPU allows for large batch sizes
    BATCH_SIZE = 2048

    # Extended training duration to allow OneCycleLR to converge properly
    NUM_EPOCHS = 35

    # Optimizer settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (OneCycleLR)
    PCT_START = 0.3

    # Early Stopping
    PATIENCE = 7

    # -------------------------------------------------------------------------
    # Data Loading & Compute
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Ensure idea_8 directory exists as per specific requirements
        os.makedirs("./working/idea_8/", exist_ok=True)

    @classmethod
    def to_dict(cls):
        """
        Returns the configuration as a dictionary.
        """
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
