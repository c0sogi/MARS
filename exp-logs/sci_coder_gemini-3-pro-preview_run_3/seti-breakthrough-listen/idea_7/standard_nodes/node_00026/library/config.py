import os
import torch


class Config:
    """
    Configuration class for the Siamese Multi-Scale Spatial-Difference Network experiment.
    """

    # --- Directories ---
    INPUT_DIR = "./input"
    OUTPUT_DIR = "./working/idea_7"
    METADATA_DIR = "./metadata"

    # Create output directory immediately
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data Paths ---
    # Using the pre-generated metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # --- Data Dimensions & Preprocessing ---
    # Original spectrogram shape is (6, 273, 256)
    # We pad height to 288 to be divisible by 32 (standard for CNN backbones)
    ORIGINAL_HEIGHT = 273
    IMG_HEIGHT = 288
    IMG_WIDTH = 256
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # --- Model Architecture ---
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    # Input channels for each stream (On-Target and Off-Target)
    # 3 'A' frames for On-Target, 3 'B/C/D' frames for Off-Target
    IN_CHANNELS = 3

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = 15  # For CosineAnnealingLR

    # Augmentation
    MIXUP_ALPHA = 0.2

    # --- Hardware & Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --- Debugging / Development ---
    # Set DEBUG to True to train on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000
