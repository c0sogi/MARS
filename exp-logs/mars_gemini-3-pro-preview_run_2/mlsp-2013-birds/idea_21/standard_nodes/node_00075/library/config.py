import os
import torch


class Config:
    """
    Configuration module for Heterogeneous Ensemble with Anchor-Based Self-Distillation.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Toggle for debugging on smaller subsets

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Source Image Directory
    # Strategy specifies using Filtered Spectrograms.
    # Since metadata CSVs point to standard spectrograms, we define this path
    # to allow the dataloader to swap the root directory.
    FILTERED_SPEC_DIR = os.path.join(
        INPUT_DIR, "supplemental_data", "filtered_spectrograms"
    )

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Resolution: 224 (Frequency/Height) x 448 (Time/Width)
    IMG_HEIGHT = 224
    IMG_WIDTH = 448
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # Input Channels: 3 (Pseudo-RGB via channel replication)
    IN_CHANNELS = 3

    # Number of bird species
    NUM_CLASSES = 19

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 35
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Augmentation
    MIXUP_ALPHA = 0.4

    # =========================================================================
    # Distillation Parameters
    # =========================================================================
    # Lambda: Weighting factor for the KL Divergence loss component
    DISTILLATION_LAMBDA = 0.5
    # Temperature: Softens the probability distributions from anchors
    DISTILLATION_TEMP = 2.0

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Anchors: Stable models trained in Stage 1 to provide soft labels
    ANCHOR_MODELS = ["resnet18", "efficientnet_b0"]

    # Student: High-capacity model stabilized via distillation in Stage 2
    STUDENT_MODEL = "densenet121"

    # Head Design: Multi-Sample Dropout rates
    DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
