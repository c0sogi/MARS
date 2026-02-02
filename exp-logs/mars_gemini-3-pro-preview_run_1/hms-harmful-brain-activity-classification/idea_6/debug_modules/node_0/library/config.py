import os
import torch


class Config:
    """
    Configuration module for the Symmetry-Aware Siamese Dual-Stream Network.
    Defines hyperparameters, file paths, and data processing constants.
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Raw Data
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata (Generated previously)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output & Caching (Idea 6 specific)
    IDEA_NAME = "idea_6"
    OUTPUT_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Data Processing Configuration
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4  # Adjust based on CPU availability (12 vCPUs available)

    # EEG Signal Parameters
    # Original is 200Hz, we downsample to 100Hz for efficiency
    EEG_SRC_SAMPLING_RATE = 200
    EEG_TARGET_SAMPLING_RATE = 100
    EEG_DURATION_SEC = 50
    EEG_SEQ_LENGTH = EEG_TARGET_SAMPLING_RATE * EEG_DURATION_SEC  # 5000 samples

    # Spectrogram Parameters
    # Resizing for the 2D CNN stream
    SPEC_RESIZE_SIZE = (512, 512)

    # Channel Mapping for Siamese Architecture
    # Splitting 10-20 system into Left/Right for symmetry analysis
    LEFT_HEMISPHERE_CHANNELS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1"]
    RIGHT_HEMISPHERE_CHANNELS = ["Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
    # Midline channels (often excluded in strict lateral comparisons or used as auxiliary)
    MIDLINE_CHANNELS = ["Fz", "Cz", "Pz"]

    # =========================================================================
    # 3. Model Hyperparameters
    # =========================================================================
    # Training
    BATCH_SIZE = 32
    EPOCHS = 8
    LEARNING_RATE = 1e-3
    MAX_LR = 5e-3  # For OneCycleLR
    WEIGHT_DECAY = 0.01
    PATIENCE = 3  # Early stopping patience

    # Architecture - Stream A (EEG Siamese)
    CNN_KERNELS = [3, 5, 7, 9]  # Multi-scale kernels
    CNN_FILTERS = 64
    CNN_DROPOUT = 0.2

    # Architecture - Stream B (Spectrogram)
    SPEC_BACKBONE = "efficientnet_b0"
    SPEC_PRETRAINED = True

    # Fusion
    ATTENTION_DIM = 256

    # =========================================================================
    # 4. Labels & Metrics
    # =========================================================================
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]
    CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
    NUM_CLASSES = 6

    # =========================================================================
    # 5. Hardware & Setup
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def init_directories(cls):
        """
        Initialize the directory structure for the current experiment.
        Creates cache, checkpoint, and submission directories.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.OUTPUT_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.SUBMISSION_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"Configuration initialized. Output directory: {cls.OUTPUT_DIR}")
