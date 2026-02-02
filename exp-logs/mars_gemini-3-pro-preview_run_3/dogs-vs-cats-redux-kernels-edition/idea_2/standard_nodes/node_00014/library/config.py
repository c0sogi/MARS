import os
import torch


class Config:
    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # System & Hardware
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 256
    NUM_CLASSES = 2  # Dog vs Cat

    # Augmentation Parameters
    AUG_CROP_SCALE = (0.8, 1.0)  # RandomResizedCrop scale
    AUG_COLOR_JITTER = 0.4  # Strength for ColorJitter (brightness, contrast, sat, hue)
    AUG_HFLIP_PROB = 0.5  # Probability of horizontal flip

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Using ConvNeXt-Tiny pre-trained on ImageNet-1k
    MODEL_NAME = "convnext_tiny.in12k_ft_in1k"
    PRETRAINED = True

    # =========================================================================
    # Training Configuration
    # =========================================================================
    EPOCHS = 7
    BATCH_SIZE = 128

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # Scheduler (Cosine Annealing)
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_ETA_MIN = 1e-6

    # Regularization
    LABEL_SMOOTHING = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    # Lower layers get lower LR: lr * (decay_rate ** depth)
    USE_LLRD = True
    LLRD_DECAY_RATE = 0.9

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Test Time Augmentation
    USE_TTA = True  # Enables averaging predictions of original and h-flipped images
