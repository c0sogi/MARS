import os
import torch


class Config:
    """
    Configuration class for the EEG Harmful Brain Activity Detection project.
    Centralizes all file paths, hyperparameters, and data settings.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Raw Data Paths
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Specifications
    # ==========================================
    SAMPLING_RATE = 200
    DURATION = 50  # seconds
    N_SAMPLES = SAMPLING_RATE * DURATION  # 10,000 samples

    # Standard 10-20 System Channels (19 channels, excluding EKG)
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
    N_CHANNELS = len(EEG_CHANNELS)

    # Target Columns (Probabilities)
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]
    NUM_CLASSES = len(TARGET_COLS)

    # ==========================================
    # Preprocessing & Spectrograms
    # ==========================================
    IMG_SIZE = (512, 512)  # (Frequency, Time) input size for the model

    # MelSpectrogram Parameters
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 20  # Small hop to maintain temporal resolution
    FMIN = 0
    FMAX = SAMPLING_RATE // 2  # Nyquist frequency (100 Hz)

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "tf_efficientnet_b2.ns_jft_in1k"  # Pretrained on ImageNet-21k
    DROPOUT = 0.2
    DROP_PATH_RATE = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0
    PATIENCE = 3  # For Early Stopping

    # Scheduler
    T_MAX = EPOCHS  # Cosine Annealing duration
    MIN_LR = 1e-6

    # ==========================================
    # Compute & Debugging
    # ==========================================
    NUM_WORKERS = os.cpu_count()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to an integer (e.g., 1000) to train on a subset for debugging, or None for full run
    DEBUG_SUBSET_SIZE = None

    @classmethod
    def print_config(cls):
        """Prints the configuration settings."""
        print(f"\n{'='*40}")
        print(f"CONFIGURATION")
        print(f"{'='*40}")
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<20}: {v}")
        print(f"{'='*40}\n")
