import os
import torch


class Config:
    """
    Configuration for Calibrated 2.5D Multi-Level Sequence Network.
    Centralizes all hyperparameters, file paths, and training settings.
    """

    # --- General Settings ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    EXP_NAME = "idea_14"
    OUTPUT_DIR = os.path.join("./working", EXP_NAME)

    # --- Data Paths ---
    DATA_ROOT = "./input"
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")

    # Metadata Paths (Generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --- Data Processing ---
    IMAGE_SIZE = (384, 384)
    IN_CHANNELS = 3  # 2.5D Stacking: Input is (Slice_z-1, Slice_z, Slice_z+1)
    SEQ_LEN = 96  # Sequence length (Z-depth) for the LSTM
    NUM_WORKERS = 4  # Number of data loading workers

    # --- Model Architecture ---
    BACKBONE = "efficientnet_b4"  # Feature extractor
    PRETRAINED = True
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.2
    USE_MULTI_LEVEL_FEATURES = True  # Aggregates features from P4 and P5 blocks

    # --- Targets & Loss ---
    NUM_CLASSES = 8
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Loss Weights (w_j in the metric formula)
    # 'patient_overall' is weighted higher (7.0) vs individual vertebrae (1.0)
    LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0]

    # Positive Class Weight for BCE
    # STRICTLY 1.0 to ensure probabilistic calibration (Lesson 00042)
    POS_WEIGHT = 1.0

    # --- Training Hyperparameters ---
    EPOCHS = 10
    # Batch size is small due to large sequence length (96) and B4 backbone
    BATCH_SIZE = 2
    # Gradient accumulation simulates a larger effective batch size (2 * 8 = 16)
    GRAD_ACCUMULATION_STEPS = 8

    LEARNING_RATE = 2e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 10.0

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures the output directory exists."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
