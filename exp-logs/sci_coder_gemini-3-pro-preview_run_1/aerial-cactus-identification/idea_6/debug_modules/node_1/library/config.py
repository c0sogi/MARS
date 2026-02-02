import os
import torch


class Config:
    """
    Centralized configuration for the Cactus Classifier project (Idea 6).
    Handles file paths, hyperparameters, and device settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    PROJECT_NAME = "Cactus_Classifier_RepVGG_DeepSup"
    DEBUG = False  # Set to True to run on a small subset for testing

    # ==========================================
    # Directory & File Paths
    # ==========================================
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (For Caching & Checkpoints)
    # Specific to Idea 6 as per instructions
    WORKING_DIR = "./working/idea_6"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Data Cache Paths (Numpy format for RAM caching)
    CACHE_TRAIN_IMGS = os.path.join(WORKING_DIR, "cache_train_imgs.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "cache_train_labels.npy")
    CACHE_VAL_IMGS = os.path.join(WORKING_DIR, "cache_val_imgs.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "cache_val_labels.npy")
    CACHE_TEST_IMGS = os.path.join(WORKING_DIR, "cache_test_imgs.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "cache_test_ids.npy")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = 32
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 30
    # A100 has 40GB VRAM, can handle large batches for 32x32 images easily.
    # Using 256 for stable gradients and speed.
    BATCH_SIZE = 256

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    # T_max will be set to EPOCHS in the training loop
    ETA_MIN = 1e-6

    # ==========================================
    # Regularization & Architecture Specifics
    # ==========================================
    # Mixup Regularization
    MIXUP_ALPHA = 0.2

    # Deep Supervision
    AUX_LOSS_WEIGHT = 0.4

    # ==========================================
    # Compute
    # ==========================================
    NUM_WORKERS = 4  # 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_summary(cls):
        """Prints a summary of the current configuration."""
        print(f"\n[Config] {cls.PROJECT_NAME}")
        print(f"  Device:          {cls.DEVICE}")
        print(f"  Epochs:          {cls.EPOCHS}")
        print(f"  Batch Size:      {cls.BATCH_SIZE}")
        print(f"  Learning Rate:   {cls.LEARNING_RATE}")
        print(f"  Mixup Alpha:     {cls.MIXUP_ALPHA}")
        print(f"  Aux Loss Weight: {cls.AUX_LOSS_WEIGHT}")
        print(f"  Working Dir:     {cls.WORKING_DIR}")
        print(f"  Submission Path: {cls.SUBMISSION_PATH}")
        print("-" * 30)
