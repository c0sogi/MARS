import os
import torch


class Config:
    # =========================================================================
    # Path Configuration
    # =========================================================================
    PROJECT_ROOT = "."
    INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
    TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
    TEST_DIR = os.path.join(INPUT_DIR, "test2")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output & Cache Paths
    # Using 'idea_6' as the current iteration workspace
    IDEA_NAME = "idea_6"
    WORKING_DIR = os.path.join(PROJECT_ROOT, "working", IDEA_NAME)
    SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Audio Processing Parameters
    # =========================================================================
    # Native sample rate is 2000Hz
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds

    # Spectrogram Parameters
    # High resolution Mels (384) requires sufficient FFT size
    # N_FFT=1024 gives 513 freq bins, enough for 384 mels
    N_MELS = 384
    N_FFT = 1024
    # Hop length of 20 samples at 2000Hz is 10ms
    HOP_LENGTH = 20
    FMIN = 20
    FMAX = 1000  # Nyquist frequency

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # EfficientNetV2-Medium with ImageNet-21k pretraining
    BACKBONE = "tf_efficientnetv2_m"
    POOLING = "GeM"  # Generalized Mean Pooling
    NUM_CLASSES = 1
    PRETRAINED = True

    # Regularization
    DROP_PATH_RATE = 0.2
    DROPOUT_RATE = 0.3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for A100 40GB and V2-M backbone
    EPOCHS = 20

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # Loss Function
    # Weighted BCE to handle 9:1 imbalance
    USE_WEIGHTED_LOSS = True

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4

    # SpecAugment
    SPEC_AUG_TIME_MASK = 20
    SPEC_AUG_FREQ_MASK = 30

    # =========================================================================
    # Compute & Debugging
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to True to run on a small subset for testing pipeline
    DEBUG = False
