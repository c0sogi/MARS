import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"

    # Input Data Paths
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 12
    LEARNING_RATE = 3e-4
    NUM_WORKERS = 4  # 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data & Signal Processing
    # ==========================================
    SAMPLING_RATE = 200  # Hz
    EEG_DURATION = 50  # Seconds
    SPEC_DURATION = 600  # Seconds (10 minutes)

    # EEG Channels
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
    N_EEG_CHANNELS = 19

    # Stream A: Physiologically-Aligned Specs
    # Target size for resizing all views before stacking
    STREAM_A_IMG_SIZE = (128, 500)  # (Freq Bins, Time Steps)

    # Configuration for the 3 distinct views
    # Window size in seconds determines the STFT n_fft (Window * SR)
    PHYSIO_VIEWS = [
        {
            "name": "slow",
            "fmin": 0.5,
            "fmax": 8.0,
            "window_size_sec": 2.0,  # Long window for freq resolution
            "hop_length_ratio": 0.1,  # overlap control
        },
        {
            "name": "fast",
            "fmin": 8.0,
            "fmax": 25.0,
            "window_size_sec": 0.1,  # Short window for temporal resolution
            "hop_length_ratio": 0.5,
        },
        {
            "name": "broadband",
            "fmin": 0.5,
            "fmax": 25.0,
            "window_size_sec": 1.0,  # Medium window for context
            "hop_length_ratio": 0.2,
        },
    ]

    # Total input channels for Stream A: 19 electrodes * 3 views = 57
    IN_CHANNELS_A = N_EEG_CHANNELS * len(PHYSIO_VIEWS)

    # Stream B: Long-Term Context Specs
    STREAM_B_IMG_SIZE = (256, 256)
    IN_CHANNELS_B = 4  # 4 anatomical regions (LL, RL, LP, RP)

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE_A = "efficientnet_b2"
    BACKBONE_B = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 6
    CLASS_NAMES = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
