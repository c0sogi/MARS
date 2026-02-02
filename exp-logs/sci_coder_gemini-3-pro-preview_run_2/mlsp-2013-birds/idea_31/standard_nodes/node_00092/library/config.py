import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True for quick debugging runs
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Image Source
    # The idea specifies using 'Filtered Spectrograms'.
    # Metadata points to 'spectrograms', so we will need to replace the parent dir in the dataset class.
    SPECTROGRAM_DIR_NAME = "filtered_spectrograms"
    BASE_IMG_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    # Output Directories
    # Specific working directory for this idea (Idea 31)
    WORKING_DIR = "./working/idea_31"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Rectangular resolution: 224 (Freq) x 448 (Time)
    IMG_HEIGHT = 224
    IMG_WIDTH = 448
    NUM_CHANNELS = 3  # Pseudo-RGB
    NUM_CLASSES = 19

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    BATCH_SIZE = 32

    # Optimization
    EPOCHS = 25 if not DEBUG else 2
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # Regularization
    MIXUP_ALPHA = 0.4
    EARLY_STOPPING_PATIENCE = 7

    # Distillation
    DISTILLATION_LAMBDA = 1.0  # Weight for the soft-target loss term

    # =========================================================================
    # Model Architecture Configuration
    # =========================================================================
    # The ensemble consists of 3 distinct backbones
    MODEL_RESNET = "resnet18"
    MODEL_EFFICIENTNET = "efficientnet_b0"
    MODEL_DENSENET = "densenet121"

    # Phase 1: Anchor Training (Stable models)
    PHASE1_MODELS = [MODEL_RESNET, MODEL_EFFICIENTNET]

    # Phase 3: Born-Again Ensemble (All models, including the harder-to-train DenseNet)
    PHASE3_MODELS = [MODEL_RESNET, MODEL_EFFICIENTNET, MODEL_DENSENET]

    # =========================================================================
    # Inference & TTA Configuration
    # =========================================================================
    # Cyclic TTA: Original + 3 Time-Shifts [0, 25%, 50%, 75% width]
    TTA_STEPS = 4

    @classmethod
    def get_spectrogram_path(cls, rel_path):
        """
        Helper to convert metadata relative path to the correct filtered spectrogram path.
        Metadata paths look like: 'supplemental_data/spectrograms/PC10_....bmp'
        We want: 'supplemental_data/filtered_spectrograms/PC10_....bmp'
        """
        # Extract filename
        filename = os.path.basename(rel_path)
        # Construct new path
        return os.path.join(cls.BASE_IMG_DIR, cls.SPECTROGRAM_DIR_NAME, filename)
