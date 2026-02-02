import os
import torch


class Config:
    """
    Configuration for WITS-Net (Weight-Inited Thick-Slab Independent Instance Network).
    Defines hyperparameters, file paths, and structural constants.
    """

    # ==========================================
    # File System & Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for caching and model artifacts
    WORKING_DIR = "./working/idea_14"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Data Pipeline Parameters
    # ==========================================
    # Modalities used in the 9-channel input
    # Order matters: FLAIR (0-2), T1wCE (3-5), T2w (6-8)
    MODALITIES = ["flair", "t1wce", "t2w"]

    # Slab Extraction
    SLAB_DEPTH = 3  # Number of consecutive slices per slab
    SLAB_STRIDE = 10  # Delta: Distance between Lower, Center, and Upper slabs
    NUM_SLABS = 3  # We extract 3 slabs (Lower, Center, Upper) per subject

    # Image Specs
    IMAGE_SIZE = 224
    IN_CHANNELS = 9  # 3 modalities * 3 slices

    # Caching
    # If True, tries to load processed tensors from .npy files in WORKING_DIR
    LOAD_CACHED_DATA = True

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 1
    PRETRAINED = True

    # Regularization
    DROPOUT_RATE = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 15  # Can be overridden during training loop
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay for small dataset

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # If set to True, pipeline should only use a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 20
