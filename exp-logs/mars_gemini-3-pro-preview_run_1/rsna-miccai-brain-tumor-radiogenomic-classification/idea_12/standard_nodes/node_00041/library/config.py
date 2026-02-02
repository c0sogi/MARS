import os
import torch


class Config:
    """
    Configuration for the Stratified Instance-Level 2.5D Network (SIL-Net) experiment.
    """

    # ==========================================
    # Experiment Control & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False
    # If DEBUG is True, limit the dataset to this many samples for quick testing
    DEBUG_DATASET_SIZE = 50

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea iteration
    WORKING_DIR = "./working/idea_12"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output
    SUBMISSION_PATH = "./submission/submission.csv"

    # Caching Paths (for deterministic data processing)
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "cache_train_images.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "cache_train_ids.npy")
    CACHE_TRAIN_TARGETS = os.path.join(WORKING_DIR, "cache_train_targets.npy")

    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "cache_val_images.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "cache_val_ids.npy")
    CACHE_VAL_TARGETS = os.path.join(WORKING_DIR, "cache_val_targets.npy")

    # ==========================================
    # Data Strategy Parameters
    # ==========================================
    IMAGE_SIZE = 224

    # Modalities used: FLAIR, T1wCE, T2w
    CHANNELS = 3

    # Instance Sampling Strategy:
    # We take the median slice (0) and slices at offsets -2 and +2.
    # This creates 3 independent training instances per subject.
    SLICE_OFFSETS = [-2, 0, 2]
    NUM_INSTANCES_PER_SUBJECT = len(SLICE_OFFSETS)

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    NUM_EPOCHS = 15

    # Optimizer settings (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Cross-Validation
    N_FOLDS = 5

    # ==========================================
    # Compute Resources
    # ==========================================
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
