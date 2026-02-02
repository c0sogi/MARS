import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_29"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Directory for processed data/features
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Spectrogram Source
    # We use 'filtered_spectrograms' as per the strategy to reduce noise.
    # Note: The metadata CSVs point to 'spectrograms'. The dataset loader
    # should replace the directory component with this path.
    SPECTROGRAM_DIR = os.path.join(
        INPUT_DIR, "supplemental_data", "filtered_spectrograms"
    )

    # Submission Output Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data & Problem Configuration
    # =========================================================================
    NUM_SPECIES = 19
    SEED = 42
    NUM_FOLDS = 5  # For Iterative Stratified K-Fold in Phase 1

    # Input Image Configuration
    # We use Pseudo-RGB (3 channels) to leverage ImageNet pretrained weights
    IN_CHANNELS = 3

    # Multi-Resolution Strategy (Height/Freq, Width/Time)
    # ResNet18 / EfficientNet-B0 (Anchors)
    IMG_SIZE_ANCHOR = (224, 448)
    # DenseNet121 (Student/Dense Bias) - Lower resolution to prevent overfitting
    IMG_SIZE_DENSENET = (160, 320)

    # =========================================================================
    # Model Architecture Configuration
    # =========================================================================
    # Phase 1: Stable Anchors
    MODELS_ANCHOR = ["resnet18", "efficientnet_b0"]

    # Phase 3: Born-Again Ensemble (Students)
    MODELS_ENSEMBLE = ["resnet18", "efficientnet_b0", "densenet121"]

    # Multi-Sample Dropout Head Parameters
    DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Augmentation
    MIXUP_ALPHA = 0.4

    # Distillation Strategy
    # Weight for the soft-target loss component
    DISTILLATION_LAMBDA = 1.0

    # =========================================================================
    # Inference & Test-Time Augmentation (TTA)
    # =========================================================================
    # Cyclic TTA: Original + 3 Time-Shifts (0.25, 0.50, 0.75 of width)
    TTA_STEPS = 4

    # =========================================================================
    # System & Debugging
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging flags
    # Set DEBUG to True to run on a small subset of data for rapid testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
