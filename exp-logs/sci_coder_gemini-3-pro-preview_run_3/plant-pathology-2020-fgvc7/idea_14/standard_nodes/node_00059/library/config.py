import os
import torch


class Config:
    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    DEBUG = False  # Set to True to run on a subset for testing

    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files (Generated Metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Images
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Output Files
    TEACHER_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_effnetv2_m.pth")
    STUDENT_CHECKPOINT = os.path.join(WORKING_DIR, "student_maxvit_small.pth")
    FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache for deterministic processing
    CACHE_DIR = WORKING_DIR

    # =========================================================================
    # Data Configuration
    # =========================================================================
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)
    CLASS_MAP = {label: i for i, label in enumerate(CLASSES)}

    # Class weights file (calculated during training initialization)
    CLASS_WEIGHTS_PATH = os.path.join(WORKING_DIR, "class_weights.npy")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone 1: EfficientNetV2-M (Teacher)
    # High-frequency texture extraction, 512x512 resolution
    TEACHER_BACKBONE = "tf_efficientnetv2_m"
    TEACHER_IMG_SIZE = 512

    # Backbone 2: EfficientNetV2-S (Student)
    # Global context via Multi-Axis Attention, 384x384 resolution
    STUDENT_BACKBONE = "tf_efficientnetv2_s"
    STUDENT_IMG_SIZE = 384

    # Feature Pyramid Network (FPN) & Head
    FPN_OUT_CHANNELS = 256
    GEM_P = 3.0  # Generalized Mean Pooling power
    DROP_RATE = 0.3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16  # Tuned for A100 40GB
    EPOCHS = 25
    PATIENCE = 10  # Relaxed patience for EMA convergence

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # Distillation
    DISTILLATION_ALPHA = 0.5  # Weight for KL Divergence loss
    TEMPERATURE = 3.0  # Softmax temperature for distillation

    # =========================================================================
    # Augmentation Strategy
    # =========================================================================
    # Strong Geometric Augmentations
    AUG_PROB = 1.0
    SHIFT_LIMIT = 0.1
    SCALE_LIMIT = 0.2
    ROTATE_LIMIT = 15

    # Probabilities
    AUG_SHIFT_SCALE_ROTATE_PROB = 0.7
    AUG_HORIZONTAL_FLIP_PROB = 0.5

    # Explicit Exclusions (Logic handled in dataset.py, documented here)
    # USE_CUTOUT = False
    # USE_COLOR_JITTER = False
    # USE_VERTICAL_FLIP = False

    # =========================================================================
    # Inference / TTA
    # =========================================================================
    TTA_ENABLED = True
    # Only Horizontal Flip allowed (preserves gravity priors)
    TTA_FLIP_HORIZONTAL = True
    TTA_FLIP_VERTICAL = False
