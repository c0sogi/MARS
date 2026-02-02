import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection Task.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure mutable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Model Configuration
    # ==========================================
    # Using ConvNeXt-Small as per strategy (Lesson 14)
    MODEL_NAME = "convnext_small.fb_in1k"
    PRETRAINED = True
    NUM_CLASSES = 6
    IMAGE_SIZE = 384  # High resolution strategy (Lesson 9)
    USE_GEM_POOLING = True  # Generalized Mean Pooling for fine-grained features

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 20
    # Batch size 32 fits ConvNeXt-Small @ 384 on A100 40GB with gradients + EMA
    # Reduced to 8 to fit on ~16GB GPU (Traceback indicated 15.77 GiB capacity)
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Scheduler (Cosine Annealing)
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_MIN_LR = 1e-6

    # Model EMA (Exponential Moving Average)
    USE_EMA = True
    # Decay 0.99 is faster/better for short training schedules (Lesson 21)
    EMA_DECAY = 0.99

    # ==========================================
    # Data Augmentation
    # ==========================================
    # Strong augmentation to prevent overfitting
    AUG_SCALE_MIN = 0.5  # Aggressive cropping (Lesson 4)
    AUG_SCALE_MAX = 1.0
    AUG_COLOR_JITTER = 0.2  # Safe lighting invariance (Lesson 12)
    AUG_HORIZONTAL_FLIP_PROB = 0.5
    AUG_VERTICAL_FLIP_PROB = 0.5  # Leaves have no fixed orientation (Lesson 11)

    # ==========================================
    # Inference
    # ==========================================
    TTA_ENABLED = True  # Test Time Augmentation
    CONF_THRESHOLD = 0.5

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12

    # ==========================================
    # Labels
    # ==========================================
    # Alphabetically sorted list of unique diseases found in analysis
    LABELS = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # Mappings
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
