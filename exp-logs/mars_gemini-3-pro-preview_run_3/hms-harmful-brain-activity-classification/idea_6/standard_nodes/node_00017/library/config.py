import os
import torch


class Config:
    """
    Configuration class for the Orthogonal Dual-Stream Network.
    Defines paths, hyperparameters, and constants for data processing and training.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs for data loading, capped at reasonable number to avoid overhead
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Output Paths
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    # EEG Stream (Stream A)
    EEG_RAW_SR = 200
    EEG_TARGET_SR = 50
    EEG_DURATION = 50
    EEG_SEQ_LEN = 2500  # 50 Hz * 50 seconds

    # Standard 19 EEG Channels (excluding EKG)
    EEG_CHANNELS = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fz",
        "Cz",
        "Pz",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "F8",
        "T4",
        "T6",
        "O2",
    ]
    N_EEG_CHANNELS = len(EEG_CHANNELS)

    # Spectrogram Stream (Stream B)
    # Input shape: (Batch, 4, 256, 256)
    SPEC_SIZE = (256, 256)  # (Height/Freq, Width/Time)
    N_SPEC_CHANNELS = 4  # LL, RL, LP, RP regions

    # Targets
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]
    N_CLASSES = len(TARGET_COLS)

    # Submission Headers
    SUBMISSION_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # ==========================================
    # Model & Training Configuration
    # ==========================================
    # Architecture
    BACKBONE_STREAM_B = "efficientnet_b0"
    PRETRAINED = True

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10

    # Differential Optimization
    # Stream A (Scratch 1D-ResNet) needs higher LR to converge
    # Stream B (Pretrained EfficientNet) needs lower LR to preserve features
    LR_STREAM_A = 1e-3
    LR_STREAM_B = 1e-4

    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0

    # Learning Rate Scheduler
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 3

    # Debugging
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000
