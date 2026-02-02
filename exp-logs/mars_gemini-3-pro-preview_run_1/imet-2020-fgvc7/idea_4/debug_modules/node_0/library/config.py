import os
import torch


class Config:
    """
    Configuration class for the Artwork Attribute Labeling task.
    Implements 'Idea 4': ConvNeXt-Small backbone with EMA, GeM Pooling, and robust training strategies.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Paths
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "convnext_small_best.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Upgrading to ConvNeXt-Small as per Idea 4
    # Using 'fb_in22k_ft_in1k' weights which are pre-trained on ImageNet-22k and fine-tuned on 1k
    MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
    PRETRAINED = True
    NUM_CLASSES = 3474
    USE_GEM_POOLING = True  # Generalized Mean Pooling

    # -------------------------------------------------------------------------
    # Input Parameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 320
    INPUT_SHAPE = (320, 320)
    CHANNELS = 3

    # ImageNet Normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # A100 40GB can handle batch size 64 for ConvNeXt-Small @ 320x320
    BATCH_SIZE = 64
    EPOCHS = 18  # Extended training duration (15-20 epochs) to ensure convergence

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Loss Function (BCEWithLogitsLoss)
    POS_WEIGHT = 12.0  # Moderate positive weight to handle class imbalance without precision loss
    LABEL_SMOOTHING = 0.05  # To mitigate noisy labels

    # -------------------------------------------------------------------------
    # Stabilization & Regularization
    # -------------------------------------------------------------------------
    # Model EMA (Exponential Moving Average)
    USE_EMA = True
    EMA_DECAY = 0.9999

    # Gradient Clipping
    MAX_GRAD_NORM = 10.0

    # -------------------------------------------------------------------------
    # Augmentation (Albumentations)
    # -------------------------------------------------------------------------
    # Aggressive augmentation strategy to prevent overfitting
    AUG_SHIFT_LIMIT = 0.1
    AUG_SCALE_LIMIT = 0.1
    AUG_ROTATE_LIMIT = 15
    AUG_COLOR_JITTER_BRIGHTNESS = 0.2
    AUG_COLOR_JITTER_CONTRAST = 0.2
    AUG_COLOR_JITTER_SATURATION = 0.2
    AUG_COLOR_JITTER_HUE = 0.1

    # -------------------------------------------------------------------------
    # Inference / Validation
    # -------------------------------------------------------------------------
    # Threshold tuning search space
    THRESHOLD_START = 0.01
    THRESHOLD_END = 0.99
    THRESHOLD_STEP = 0.01
