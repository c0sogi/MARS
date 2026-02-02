import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Defines file paths, data constants, model hyperparameters, and training settings.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Input Metadata Paths (Parquet files)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Processed Data Cache Paths (Numpy files)
    # Using .npy for efficient loading of preprocessed tensors
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npy")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npy")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npy")

    # Output Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Dimensions and Features
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Feature Channels Configuration
    # Sequence: A, G, C, U -> 4 channels
    # Structure: (, ), . -> 3 channels
    # Loop Type: S, M, I, B, H, E, X -> 7 channels
    NUM_CHANNELS = 4 + 3 + 7  # Total: 14

    # Target Definition
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Columns used for the competition metric (MCRMSE)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices of scored columns within the TARGET_COLS list
    SCORED_COLS_INDICES = [0, 1, 3]

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Dilated Residual CNN (Deprecated in favor of larger RNN backbone)
    CNN_FILTERS = 256
    CNN_KERNEL_SIZE = 3
    # Geometrically increasing dilation rates to capture multi-scale motifs
    DILATION_RATES = [1]  # Simplified to single layer

    # Recurrent Backbone (BiGRU)
    # Increased capacity based on Lesson 00026
    RNN_HIDDEN_DIM = 256
    RNN_LAYERS = 2

    # Regularization
    DROPOUT = 0.4

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    NUM_WORKERS = 2
