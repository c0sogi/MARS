import os
import torch


class Config:
    # Random Seed
    SEED = 42

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    WORKING_DIR = "./working/idea_4"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Architecture
    # Using ConvNeXt-Small pretrained on ImageNet-22k and fine-tuned on ImageNet-1k
    MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
    IMG_SIZE = 384
    NUM_CLASSES = 1  # Binary classification (Dog vs Cat)

    # Training Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 32  # Adjusted for 384x384 resolution on A100
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0

    # Augmentation
    MIXUP_ALPHA = 0.4
    CROP_SCALE_MIN = 0.8
    CROP_SCALE_MAX = 1.0

    # Exponential Moving Average (EMA)
    USE_EMA = True
    EMA_DECAY = 0.9999

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Debugging
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000  # Number of samples to use when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
