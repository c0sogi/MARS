import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for Idea 19 (Single View, Independent Anchors)
    WORKING_DIR = "./working/idea_19"
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 224
    NUM_SLICES = 3  # Number of slices per modality
    STRIDE = 5  # Stride between slices
    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

    # Input channels = Modalities * Slices (4 * 3 = 12)
    IN_CHANNELS = NUM_MODALITIES * NUM_SLICES

    # Depth filtering for ROI selection (15% - 85%)
    DEPTH_MIN = 0.15
    DEPTH_MAX = 0.85

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    DROPOUT_RATE = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 15

    # Optimizer settings
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Augmentation
    ROTATION_DEGREES = 15

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Caching Filenames
    # --------------------------------------------------------------------------
    # We use .npy for image arrays and .parquet for metadata/labels to avoid pickle
    CACHE_TRAIN_DATA = os.path.join(CACHE_DIR, "train_data.npy")
    CACHE_TRAIN_LABELS = os.path.join(CACHE_DIR, "train_labels.npy")

    CACHE_VAL_DATA = os.path.join(CACHE_DIR, "val_data.npy")
    CACHE_VAL_LABELS = os.path.join(CACHE_DIR, "val_labels.npy")

    CACHE_TEST_DATA = os.path.join(CACHE_DIR, "test_data.npy")
    CACHE_TEST_IDS = os.path.join(CACHE_DIR, "test_ids.npy")
