import os
import torch


class Config:
    """
    Configuration for the Tri-View Hierarchical Fusion Network.
    Centralizes paths, data parameters, and training hyperparameters.
    """

    # =========================================================================
    # File System & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure working and cache directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Source Paths
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    OUTPUT_DIR = WORKING_DIR
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = "./submission/submission.csv"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Data Specifications (Tri-View)
    # =========================================================================
    SEED = 42
    NUM_CLASSES = 6

    # Target Columns (Probabilities)
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # --- View A: Micro (Raw EEG) ---
    # Captures fine-grained waveform dynamics
    EEG_RAW_SAMPLING_RATE = 200
    EEG_TARGET_SAMPLING_RATE = 100  # Downsample for efficiency
    EEG_DURATION_SECONDS = 50
    EEG_SEQ_LEN = EEG_DURATION_SECONDS * EEG_TARGET_SAMPLING_RATE  # 5000 time steps

    # EEG Channels (Standard 10-20 system + EKG)
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
        "EKG",
    ]

    # --- View B: Meso (Local Spectrogram) ---
    # Captures high-res time-frequency features of the specific event
    # 50-second crop aligned with EEG
    MESO_IMG_SIZE = (224, 224)  # (Height, Width)

    # --- View C: Macro (Global Spectrogram) ---
    # Captures long-range context (10 minutes)
    MACRO_IMG_SIZE = (512, 512)  # (Height, Width)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # Optimization
    BATCH_SIZE = 32
    EPOCHS = 12
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Scheduler (OneCycleLR)
    PCT_START = 0.1
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Early Stopping
    PATIENCE = 4

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 1000
