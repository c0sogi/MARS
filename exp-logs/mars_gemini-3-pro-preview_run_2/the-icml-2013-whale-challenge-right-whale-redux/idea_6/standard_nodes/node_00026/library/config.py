import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    PROJECT_NAME = "RightWhaleDetection_Ensemble"
    SEED = 42
    NUM_WORKERS = 4  # Optimized for the available 12 vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")

    # Working Directory (for Caching and Checkpoints)
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    OUTPUT_SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(OUTPUT_SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    # Strategy: Single-Channel Log-Mel Spectrograms
    SR = 2000  # Sampling rate
    DURATION = 2.0  # Target duration in seconds
    N_FFT = 1024  # Large window for frequency resolution
    HOP_LENGTH = 64  # Small hop for high temporal resolution
    N_MELS = 128  # Number of Mel bands
    FMIN = 0  # Min frequency
    FMAX = None  # Max frequency (None -> SR/2)
    NORMALIZED_MEL = False  # False to preserve spectral tilt of Pink noise

    # ==========================================
    # Model Architecture
    # ==========================================
    # Heterogeneous Ensemble Strategy
    MODEL_NAMES = ["efficientnet_b0", "resnet34"]
    IN_CHANNELS = 1  # Modified first layer for spectrograms
    NUM_CLASSES = 1  # Binary classification
    PRETRAINED = True  # Use ImageNet weights

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128  # Maximize for batch norm stability
    EPOCHS = 15  # Sufficient for convergence with early stopping
    LEARNING_RATE = 1e-3  # Standard starting LR for AdamW
    WEIGHT_DECAY = 1e-4  # Regularization
    PATIENCE = 4  # Early stopping patience

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use in debug mode
