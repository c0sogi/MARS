import os
import torch


class Config:
    """
    Central configuration for the Robust Adversarial DeBERTa-v3-Large Ensemble.
    """

    # ==========================================
    # General Settings
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use in debug mode

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Sub-directories for artifacts
    OUTPUT_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission_null.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LEN = 160
    DROPOUT = 0.2
    NUM_CLASSES = 1

    # Freezing strategy: Embeddings + Bottom N layers
    FREEZE_LAYERS = 6

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEEDS = [42, 43, 44]
    EPOCHS = 4

    # Batch Size Configuration
    # Effective Batch Size = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
    # 4 * 8 = 32
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 8

    # Optimizer
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1

    # ==========================================
    # Adversarial Weight Perturbation (AWP)
    # ==========================================
    USE_AWP = True
    AWP_LR = 1e-4
    AWP_EPS = 1e-2
    # Start AWP after this many epochs (can be float, e.g., 0.5)
    AWP_START_EPOCH = 1.0

    # ==========================================
    # System / Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = True
