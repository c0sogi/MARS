import os
import torch


class Config:
    """
    Central configuration for the EEG Seizure Detection task.
    Stores file paths, hyperparameters, and model specifications.
    """

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure working and output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "gru_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    SEED = 42

    # Sampling Rates
    ORIGINAL_SAMPLING_RATE = 200
    TARGET_SAMPLING_RATE = 50  # Downsample to 50 Hz as per Idea

    # Temporal Dimensions
    DURATION = 50  # Seconds
    # Sequence length: 50s * 50Hz = 2500 steps
    SEQ_LENGTH = int(DURATION * TARGET_SAMPLING_RATE)

    # EEG Channels (Standard 10-20 system)
    # We focus on the 19 EEG channels, excluding EKG
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
    N_CHANNELS = len(EEG_CHANNELS)  # 19

    # Target Columns (Probabilities for training - KL Divergence targets)
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # Submission Columns (Header for final submission file)
    SUBMISSION_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    NUM_CLASSES = len(TARGET_COLS)

    # ==========================================
    # Model Architecture (Bi-GRU + Attention)
    # ==========================================
    PROJ_DIM = 12  # Dimension after initial 1x1 Conv projection
    HIDDEN_DIM = 128  # Hidden dimension for GRU
    NUM_LAYERS = 2  # Number of stacked GRU layers
    DROPOUT = 0.1  # Dropout rate

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    PATIENCE = 4  # Early stopping patience

    # Hardware
    NUM_WORKERS = 4  # Number of dataloader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Execution Flags
    # ==========================================
    # If True, runs on a small subset of data for debugging purposes
    DEBUG = False

    # If True, attempts to load pre-processed tensors from WORKING_DIR
    LOAD_CACHED_DATA = False
