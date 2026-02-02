import os
import torch


class Config:
    """
    Configuration for the Weight-Inflated Independent-Slab Network (WIIS-Net).
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic data processing)
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")

    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "val_ids.npy")

    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    IMAGE_SIZE = 224

    # Modalities to use (Order matters for channel mapping)
    # Channels 0-2: FLAIR, 3-5: T1wCE, 6-8: T2w
    MODALITIES = ["FLAIR", "T1wCE", "T2w"]

    # Slab Configuration
    SLAB_DEPTH = 3  # Number of slices per modality per slab (z-1, z, z+1)
    INPUT_CHANNELS = 9  # len(MODALITIES) * SLAB_DEPTH

    # Volumetric Expansion
    SLAB_STRIDE = 5  # Delta for expansion (Median-delta, Median, Median+delta)
    NUM_SLABS_PER_SUBJECT = 3

    # ==========================================
    # Model Parameters
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    DROPOUT_RATE = 0.3
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay for small dataset
    NUM_FOLDS = 5
    EARLY_STOPPING_PATIENCE = 3

    # Compute
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for testing pipeline
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50
