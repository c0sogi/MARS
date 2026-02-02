import os
import torch


class Config:
    """
    Configuration class for the Attentive Dual-Scale Fusion Network.
    Contains paths, signal processing parameters, model architecture settings,
    and training hyperparameters.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # Number of dataloader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TRAIN_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Output
    OUTPUT_DIR = WORKING_DIR
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Signal Processing (Stream A: EEG -> MelSpec)
    # ==========================================
    SR = 200  # Sampling Rate (Hz)
    DURATION = 50  # Duration of EEG clip (Seconds)
    N_MELS = 128  # Number of Mel bands
    FMIN = 0  # Minimum frequency (Hz)
    FMAX = 20  # Maximum frequency (Hz) - Focus on lower bands
    N_FFT = 1024  # FFT window size
    # Hop length calculated to yield approx 256 time steps for 50s @ 200Hz
    # 10000 samples / 256 steps ~= 39
    HOP_LENGTH = 39

    # ==========================================
    # Pre-computed Spectrograms (Stream B)
    # ==========================================
    SPEC_DURATION = 600  # 10 minutes (Seconds)
    SPEC_SIZE = (256, 256)  # Resize dimensions for Stream B input

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE_A = "tf_efficientnet_b2"  # For fine-grained EEG MelSpecs
    BACKBONE_B = "tf_efficientnet_b0"  # For long-term context Spectrograms
    PRETRAINED = True
    NUM_CLASSES = 6

    # Input Channels
    IN_CHANNELS_A = 19  # 19 EEG Electrodes
    IN_CHANNELS_B = 4  # 4 Regions (LL, RL, LP, RP)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 6
    LR = 3e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0
    PATIENCE = 3  # Early stopping patience

    # ==========================================
    # Data Definitions
    # ==========================================
    # Standard 10-20 EEG Montage (excluding EKG)
    EEG_NAMES = [
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

    # Target Class Columns
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]


# Ensure the working directory exists for caching and outputs
os.makedirs(Config.WORKING_DIR, exist_ok=True)
