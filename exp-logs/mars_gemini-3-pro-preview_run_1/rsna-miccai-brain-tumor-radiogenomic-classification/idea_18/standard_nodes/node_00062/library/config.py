import os
import torch


class Config:
    """
    Configuration for the Weight-Inflated Independent-Slab (WIIS) Network.
    Stores all hyperparameters, constants, and path definitions.
    """

    # ==========================================
    # Reproducibility & Debugging
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20  # Number of subjects to process if DEBUG is True

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for this specific experiment (Idea 18)
    WORKING_DIR = "./working/idea_18"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Checkpoint path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing & Structure
    # ==========================================
    IMG_SIZE = 224

    # Modalities to use in the specific order: FLAIR (0-2), T1wCE (3-5), T2w (6-8)
    # T1w is excluded based on the WIIS strategy
    SELECTED_MODALITIES = ["FLAIR", "T1wCE", "T2w"]

    # Slab configuration: Single middle slice
    NUM_SLICES_PER_MODALITY = 1

    # Total input channels = 3 modalities * 1 slice = 3 channels
    IN_CHANNELS = len(SELECTED_MODALITIES) * NUM_SLICES_PER_MODALITY

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    DROPOUT_RATE = 0.3
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 4

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Caching Configuration
    # ==========================================
    # Flag to control cache loading behavior
    LOAD_CACHED_DATA = True

    # Cache File Paths (Numpy format)
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")

    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "val_ids.npy")

    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")
