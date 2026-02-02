import os
import torch


class Config:
    """
    Configuration class for the MS-TCN Gesture Recognition project.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata CSV paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache paths for processed features (using .npz for numpy arrays)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.npz")

    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "mstcn_best_model.pth")

    # Submission output path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    RANDOM_SEED = 42

    # Skeleton configuration
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3  # X, Y, Z

    # Audio configuration
    N_MFCC = 13
    AUDIO_SAMPLE_RATE = 16000

    # Input Feature Dimension Calculation
    # Features = (Skeleton Pos) + (Skeleton Vel) + (Audio MFCC)
    #          = (20 * 3)     + (20 * 3)     + 13
    #          = 60           + 60           + 13 = 133
    INPUT_DIM = (NUM_JOINTS * COORDS_PER_JOINT * 2) + N_MFCC

    # Class Labels
    # 20 gestures (IDs 1-20) + 1 background class (ID 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # Model Hyperparameters (MS-TCN)
    # ==========================================
    NUM_STAGES = 2  # Stage 1: Prediction, Stage 2: Refinement
    NUM_LAYERS = 10  # Number of dilated residual layers per stage
    NUM_F_MAPS = 64  # Number of feature maps (channels) in hidden layers
    KERNEL_SIZE = 3  # Convolution kernel size
    DROPOUT = 0.5  # Dropout probability

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Smaller batch size due to variable sequence lengths
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 50

    # Loss Weights
    # Total Loss = CrossEntropy + LAMBDA_SMOOTH * TruncatedMSE
    LAMBDA_SMOOTH = 0.15  # Weight for temporal smoothing loss

    # Optimization
    EARLY_STOPPING_PATIENCE = 10
    GRADIENT_CLIP = 5.0  # Clip gradients to prevent explosion

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to an integer (e.g., 20) to train/val on a small subset for debugging.
    # Set to None for full training.
    DEBUG_SUBSET_SIZE = None
