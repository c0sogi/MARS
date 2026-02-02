import os
import torch


class Config:
    """
    Central configuration for the NFL Player Contact Detection task.
    Implements settings for the Center-Focused Temporal Convolutional Network (CF-TCN).
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Create necessary writable directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Processing & Feature Engineering
    # =========================================================================
    # Temporal Window Configuration
    # The model uses a window of frames centered on the target frame.
    # HALF_WINDOW_SIZE = 5 means: t-5, ..., t, ..., t+5 (Total 11 frames)
    HALF_WINDOW_SIZE = 5
    WINDOW_SIZE = 2 * HALF_WINDOW_SIZE + 1

    # Raw columns to load from tracking data
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Feature definitions per timestep
    # These features are constructed for every step in the temporal window.
    # Ground imputation logic (P2=P1) ensures geometric consistency for 'is_ground'.
    FEATURES_PER_STEP = [
        # Player 1 Kinematics
        "x_position_1",
        "y_position_1",
        "speed_1",
        "direction_1",
        "orientation_1",
        "acceleration_1",
        "sa_1",
        # Player 2 (or Imputed Ground) Kinematics
        "x_position_2",
        "y_position_2",
        "speed_2",
        "direction_2",
        "orientation_2",
        "acceleration_2",
        "sa_2",
        # Pairwise Interaction Features
        "distance",  # Euclidean distance
        "log_distance",  # np.log1p(distance) for resolution near 0
        "closing_speed",  # Rate of distance change (clamped)
        "is_ground",  # Binary flag: 1 if P2 is Ground, else 0
    ]

    NUM_FEATURES_PER_STEP = len(FEATURES_PER_STEP)

    # The "Wide" input dimension: (Features * Window_Size)
    # This is the size of the flattened vector input to the model before internal reshaping
    INPUT_WIDTH = NUM_FEATURES_PER_STEP * WINDOW_SIZE

    # =========================================================================
    # Model Architecture (CF-TCN)
    # =========================================================================
    # Convolutional Encoder
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3
    CNN_LAYERS = 3  # Number of residual 1D convolution blocks

    # Classification Head
    HIDDEN_DIM = 256  # Size of dense layers after flattening/concatenation
    DROPOUT = 0.1  # Dropout rate for regularization

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Optimization
    BATCH_SIZE = 4096  # Large batch size supported by A100 for tabular/1D data
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # Focal Loss Parameters
    # Designed to handle class imbalance (Contact ~1.4%) without discarding data
    FOCAL_ALPHA = 0.75  # Penalize false negatives more heavily
    FOCAL_GAMMA = 2.0  # Focus on hard examples

    # =========================================================================
    # Compute & Inference
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # Inference Threshold Optimization
    # Range of thresholds to search for maximizing MCC on validation set
    THRESHOLD_SEARCH_START = 0.1
    THRESHOLD_SEARCH_END = 0.6
    THRESHOLD_SEARCH_STEP = 0.01
