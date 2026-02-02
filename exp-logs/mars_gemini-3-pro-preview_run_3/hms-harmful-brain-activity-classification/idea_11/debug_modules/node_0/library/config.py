import os
import torch


class Config:
    """
    Configuration class for Band-Adaptive Multi-Resolution Network with Temporal Attention.
    Encapsulates all file paths, hyperparameters, and model settings.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files (Generated in previous steps)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Raw Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # EEG Signal Props
    EEG_SR = 200
    EEG_DURATION = 50
    EEG_SAMPLES = 10000  # 50s * 200Hz

    # EEG Channels (19 electrodes, excluding EKG)
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

    # Stream A: Band-Adaptive STFT Configurations
    # Defines the 3 bands for the Multi-Resolution input
    STFT_BANDS = [
        {
            "name": "delta_theta",
            "window_sec": 2.0,  # Long window for low freq resolution
            "hop_sec": 0.05,  # Overlap
            "fmin": 0.5,
            "fmax": 8.0,
            "n_mels": 32,  # Height of this band in the stack
        },
        {
            "name": "alpha_beta",
            "window_sec": 0.2,  # Short window for high time resolution
            "hop_sec": 0.05,
            "fmin": 8.0,
            "fmax": 25.0,
            "n_mels": 32,
        },
        {
            "name": "broadband",
            "window_sec": 1.0,  # Balanced window
            "hop_sec": 0.05,
            "fmin": 0.5,
            "fmax": 25.0,
            "n_mels": 32,
        },
    ]

    # Image Sizes for Model Input
    # Stream A: (Freq, Time) - Time dim depends on hop_sec, we resize to fixed size
    IMG_SIZE_A = (
        128,
        256,
    )  # (Height/Freq, Width/Time) - Height approx 32*3=96, padded/resized to 128

    # Stream B: 10m Spectrograms
    IMG_SIZE_B = (256, 256)  # As per description

    # ==========================================
    # Model Architecture
    # ==========================================
    # Stream A (EEG Morphological Encoder)
    BACKBONE_A = "tf_efficientnet_b2.ns_jft_in1k"
    IN_CHANNELS_A = 57  # 19 electrodes * 3 bands
    PROJ_CHANNELS_A = 3  # Projected depth before backbone

    # Stream B (Long-term Context Encoder)
    BACKBONE_B = "tf_efficientnet_b0.ns_jft_in1k"
    IN_CHANNELS_B = 4  # 4 regions (LL, RL, LP, RP) stacked depth-wise

    # Classification Head
    NUM_CLASSES = 6
    DROPOUT = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 12
    LR = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2
    PATIENCE = 4  # Early stopping patience
    MAX_GRAD_NORM = 10.0
    USE_AMP = True  # Automatic Mixed Precision

    # Augmentation
    MIXUP_ALPHA = 0.4
    SPECAUG_MASK_TIME = 20
    SPECAUG_MASK_FREQ = 10

    # ==========================================
    # Targets & Labels
    # ==========================================
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # ==========================================
    # Caching Logic
    # ==========================================
    # Paths for caching processed datasets to avoid re-computing STFTs
    CACHE_FILES = {
        "train_eeg": os.path.join(WORKING_DIR, "train_eeg_cache.npy"),
        "val_eeg": os.path.join(WORKING_DIR, "val_eeg_cache.npy"),
        "test_eeg": os.path.join(WORKING_DIR, "test_eeg_cache.npy"),
        "train_spec": os.path.join(WORKING_DIR, "train_spec_cache.npy"),
        "val_spec": os.path.join(WORKING_DIR, "val_spec_cache.npy"),
        "test_spec": os.path.join(WORKING_DIR, "test_spec_cache.npy"),
    }
