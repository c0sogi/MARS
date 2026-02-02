import os
import torch


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output and Cache paths
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_4")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Data Preprocessing ---
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 300
    WINDOW_WIDTH = 2000

    # Input Dimensions
    IMAGE_SIZE = 256
    IN_CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)

    # Sequence / Volume Parameters
    NUM_SLICES = 64  # Fixed sequence length for MIL

    # --- Model Architecture ---
    BACKBONE = "resnet18"
    NUM_CLASSES = 7  # C1-C7 (patient_overall is derived)
    HIDDEN_DIM = 512  # Feature dimension from ResNet18

    # --- Training Hyperparameters ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    # T_max set to 1.5x epochs as per strategy
    T_MAX_MULT = 1.5
    MIN_LR = 1e-6

    # Workers
    NUM_WORKERS = 4

    # --- Debugging / Development ---
    # Set to a small integer (e.g., 100) to limit dataset size for quick testing
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None
