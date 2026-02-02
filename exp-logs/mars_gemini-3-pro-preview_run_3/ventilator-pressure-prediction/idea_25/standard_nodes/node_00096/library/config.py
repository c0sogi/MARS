import os
import torch


class Config:
    """
    Central configuration for the Momentum-Corrected Residual-Hybrid Network (MCRH-Net).
    Defines hyperparameters, file paths, and model architecture specifications.
    """

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"
    SUBMISSION_DIR = "./submission"

    # Input Data (Pre-split Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "mcrh_net.pth")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (for deterministic data processing)
    CACHE_TRAIN_DATA = os.path.join(WORKING_DIR, "train_data.npy")
    CACHE_TRAIN_TARGETS = os.path.join(WORKING_DIR, "train_targets.npy")
    CACHE_VAL_DATA = os.path.join(WORKING_DIR, "val_data.npy")
    CACHE_VAL_TARGETS = os.path.join(WORKING_DIR, "val_targets.npy")
    CACHE_TEST_DATA = os.path.join(WORKING_DIR, "test_data.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Development
    # Set DEBUG to True to train on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of breaths to use in debug mode

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Strict adherence to Idea 25 specifications
    BATCH_SIZE = 128
    EPOCHS = 80
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay for unscaled regression targets
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    PATIENCE = 15  # Early stopping patience

    # ==========================================
    # Model Architecture (MCRH-Net)
    # ==========================================
    # Branch 1: Deep Residual Dense TCN (Resistive Stream)
    CNN_KERNEL_SIZE = 9
    CNN_DILATION = 1  # Dense convolutions (no dilation) for local fidelity
    CNN_CHANNELS = [64, 128, 256, 512]
    CNN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bi-LSTM (Elastic Stream)
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 3
    LSTM_DROPOUT = 0.1

    # Fusion Head
    FUSION_HIDDEN_SIZE = 1024

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================
    # Kinematics
    USE_LAG_FEATURES = True  # Backward Velocity (Momentum)
    USE_LEAD_FEATURES = True  # Forward Lookahead (Intent)
    LEAD_STEPS = 4  # u_in(t+1)...u_in(t+4)

    # Physics
    USE_PHYSICS_FEATURES = True  # Area, R*u_in, Area/C, dt

    # Sequence Information
    SEQ_LEN = 80  # Fixed breath length in dataset

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for working files and submissions.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("MCRH-Net Configuration")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("=" * 30)


# Automatically setup directories when imported
Config.setup()
