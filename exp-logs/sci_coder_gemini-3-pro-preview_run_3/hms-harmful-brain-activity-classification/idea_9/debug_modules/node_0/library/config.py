import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "idea_9")
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # General
    SEED = 42
    NUM_CLASSES = 6
    CLASS_NAMES = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # EEG Signal Parameters
    EEG_SR = 200  # Sampling rate in Hz
    EEG_DURATION = 50  # Seconds
    EEG_SAMPLES = EEG_SR * EEG_DURATION  # 10,000 samples
    N_EEG_CHANNELS = 19

    # Anatomical Chains (10-20 System) for Siamese inputs
    # LL: Left Lateral, RL: Right Lateral, LP: Left Parasagittal, RP: Right Parasagittal
    CHAIN_CONFIG = {
        "LL": ["Fp1", "F7", "T3", "T5", "O1"],
        "RL": ["Fp2", "F8", "T4", "T6", "O2"],
        "LP": ["Fp1", "F3", "C3", "P3", "O1"],
        "RP": ["Fp2", "F4", "C4", "P4", "O2"],
    }

    # EEG Spectrogram Generation (MelSpec)
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 78  # Approx 10000/128 to get ~128 time steps, or adjust for 256
    # We want output shape roughly (128, 256) or (256, 256)

    # Kaggle Spectrogram Parameters
    SPEC_DURATION = 600  # 10 minutes

    # Image Dimensions for Model Input
    # Stream A (EEG Siamese): (Batch, 4, 128, 256) -> 4 views of (128, 256)
    # Stream B (Kaggle Spec): (Batch, 1, 256, 256) -> Resized to 256x256
    IMG_SIZE = (256, 256)

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "tf_efficientnet_b0.ns_jft_in1k"
    PRETRAINED = True
    DROP_RATE = 0.2
    DROP_PATH_RATE = 0.2
    USE_SIAMESE = True

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 32  # Adjusted for A100 memory with dual streams
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 3

    # Hardware
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 1000
