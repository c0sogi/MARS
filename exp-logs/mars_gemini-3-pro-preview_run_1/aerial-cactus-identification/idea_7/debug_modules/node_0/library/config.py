import os
import torch


class Config:
    """
    Centralized configuration for the Cactus Identification task.
    Implements parameters for the Custom RepVGG + SWA pipeline.
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea iteration
    WORK_DIR = "./working/idea_7"

    # Ensure the working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # =========================================================================
    # File Paths
    # =========================================================================
    # Metadata CSVs
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (using .npy for efficient RAM loading)
    CACHE_TRAIN_IMGS = os.path.join(WORK_DIR, "cache_train_imgs.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORK_DIR, "cache_train_labels.npy")

    CACHE_VAL_IMGS = os.path.join(WORK_DIR, "cache_val_imgs.npy")
    CACHE_VAL_LABELS = os.path.join(WORK_DIR, "cache_val_labels.npy")

    CACHE_TEST_IMGS = os.path.join(WORK_DIR, "cache_test_imgs.npy")
    CACHE_TEST_IDS = os.path.join(WORK_DIR, "cache_test_ids.npy")

    # Model Checkpoints
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    FINAL_SWA_MODEL_PATH = os.path.join(WORK_DIR, "final_swa_model.pth")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    IMAGE_SIZE = 32
    NUM_CLASSES = 1  # Binary classification (has_cactus)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 256  # Optimized for A100

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler & Epochs
    EPOCHS = 50

    # Stochastic Weight Averaging (SWA)
    # Start SWA late in training to average weights around the minimum
    SWA_START_EPOCH = 35
    SWA_LR = 5e-4

    # Regularization
    MIXUP_ALPHA = 0.2  # Mild mixup as per strategy

    # =========================================================================
    # Hardware & System
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available; 4 workers is typically efficient without overhead
    NUM_WORKERS = 4
