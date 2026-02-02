import os
import torch


class Config:
    """
    Configuration for Mean-Variance Pooled ResNet34 U-Net with Stochastic Depth.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this experimental idea
    WORKING_DIR = "./working/idea_18"
    OUTPUT_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Data Loading & Preprocessing
    # =========================================================================
    IMG_SIZE = (512, 512)
    NUM_WORKERS = 12  # Utilizing available vCPUs
    PIN_MEMORY = True

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "resnet34"
    PRETRAINED = True
    IN_CHANNELS = 3

    # Study Classification Head
    NUM_STUDY_CLASSES = 4
    # Pooling strategy: 'mean', 'max', 'gem', or 'mean_std' (Dual-Pooling)
    POOLING_HEAD = "mean_std"

    # Segmentation Decoder
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    # =========================================================================
    # Regularization
    # =========================================================================
    # Stochastic Depth (DropPath) rate for the backbone
    DROP_PATH_RATE = 0.2
    # Dropout for the classification head linear layer
    DROPOUT_RATE = 0.2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 20
    BATCH_SIZE = 32

    # Optimizer (AdamW)
    # Linear Scaling Rule: Base ~1e-4 for BS=16 -> ~2e-4 for BS=32.
    # We use a robust 1e-3 for AdamW with Cosine Schedule.
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    MIN_LR = 1e-6

    # Loss Weights
    # 1:10 ratio to prioritize segmentation (dense prediction)
    LOSS_WEIGHT_CLS = 1.0
    LOSS_WEIGHT_SEG = 10.0

    # =========================================================================
    # Augmentation
    # =========================================================================
    # CoarseDropout settings to force global context learning
    AUG_COARSE_DROPOUT_PROB = 0.5
    AUG_COARSE_DROPOUT_HOLES = 8
    AUG_COARSE_DROPOUT_SIZE = int(512 * 0.1)  # approx 51px

    # Consistency Constraint: Mask must be 0 where image is dropped out
    MASK_FILL_VALUE = 0

    # MixUp is strictly excluded
    USE_MIXUP = False

    # =========================================================================
    # Inference
    # =========================================================================
    # Test-Time Augmentation
    TTA_FLIP = True

    # Gating Thresholds
    # If study prediction is 'Negative', force image prediction to 'none'
    GATING_STRATEGY = True

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
