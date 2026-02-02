import os
import torch


class Config:
    """
    Configuration for Right Whale Detection Task (Idea 5).
    Implements ConvNeXt-Small backbone with Coordinate Attention, GeM Pooling,
    and High-Resolution Mel Spectrograms.
    """

    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    PROJECT_NAME = "RightWhale_ConvNeXt_Small_Idea5"
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")

    # Working Directory for Caching and Outputs
    # Using 'idea_5' as the designated workspace for this iteration
    WORK_DIR = "./working/idea_5"
    os.makedirs(WORK_DIR, exist_ok=True)

    CACHE_DIR = WORK_DIR  # Storage for processed .npy/.parquet files
    MODEL_CHECKPOINT_DIR = WORK_DIR  # Storage for best_model.pth

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Audio Preprocessing (Ultra-High-Resolution)
    # --------------------------------------------------------------------------
    SR = 2000  # Sample Rate (Hz) - Dataset is low frequency
    DURATION = 2.0  # Clip Duration (seconds)

    # Spectrogram Parameters
    N_MELS = 320  # Vertical Resolution: High count for fine spectral detail

    # Temporal Resolution
    # 10ms hop length at 2000Hz = 20 samples.
    # Preserves rapid temporal dynamics of whale chirps.
    HOP_LENGTH = 20

    # Frequency Resolution
    # N_FFT=1024 gives 513 freq bins for 0-1000Hz.
    # Essential to support 320 Mel bands without oversampling artifacts.
    N_FFT = 1024

    FMIN = 10  # Min Frequency (Hz)
    FMAX = 1000  # Max Frequency (Hz) - Nyquist limit

    # Resulting Image Dimensions for CNN
    # Width = (2.0 * 2000) / 20 = 200 time steps
    # Height = 320 mel bands
    IMG_SIZE = (320, 200)  # (H, W)

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    BACKBONE = "convnext_small.in12k_ft_in1k"  # Increased capacity from Tiny
    PRETRAINED = True
    IN_CHANNELS = 3  # Replicating mono spectrogram to 3 channels
    NUM_CLASSES = 1

    # Advanced Architectural Components
    POOLING = "gem"  # Generalized Mean Pooling (Trainable)
    USE_COORD_ATTN = True  # Coordinate Attention blocks
    USE_MS_DROPOUT = True  # Multi-Sample Dropout
    DROPOUT_RATE = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32  # Adjusted for ConvNeXt-Small on A100
    EPOCHS = 15  # Max epochs (Early stopping usually triggers sooner)
    LEARNING_RATE = 1e-4  # Initial LR for fine-tuning
    WEIGHT_DECAY = 0.01

    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 1

    # Loss Function
    LOSS_FN = "BCEWithLogitsLoss"
    USE_CLASS_WEIGHTS = True  # Weight by inverse class frequency (approx 9:1)

    # --------------------------------------------------------------------------
    # Augmentation & Regularization
    # --------------------------------------------------------------------------
    # Mixup with Mixed Losses
    MIXUP = True
    MIXUP_ALPHA = 0.4

    # SpecAugment
    SPECAUG = True
    SPECAUG_TIME_MASK = 20  # Max time mask width (10% of width)
    SPECAUG_FREQ_MASK = 30  # Max freq mask width (10% of height)

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use in debug mode
