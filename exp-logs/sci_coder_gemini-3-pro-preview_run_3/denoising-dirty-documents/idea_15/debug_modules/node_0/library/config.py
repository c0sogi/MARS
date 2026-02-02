import os
import torch


class Config:
    """
    Configuration for High-Capacity Zero-Initialized Deep Residual Ensemble (HC-ZI-ResDnCNN).
    Defines hyperparameters, file paths, and hardware settings.
    """

    # =========================================
    # File Paths & Directories
    # =========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    CLEAN_DIR = os.path.join(INPUT_DIR, "train_cleaned")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data and saving models
    WORKING_DIR = "./working/idea_15"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================
    # Data Strategy (High-Density)
    # =========================================
    # Patch extraction settings
    PATCH_SIZE = 50
    STRIDE = 10  # Low stride for high density/overlap

    # Data Augmentation (Geometric)
    AUGMENTATION = True

    # Caching Filenames (saved in WORKING_DIR)
    CACHE_TRAIN_PATCHES = "train_patches_dense.npy"
    CACHE_TRAIN_TARGETS = "train_targets_dense.npy"
    CACHE_VAL_PATCHES = "val_patches.npy"
    CACHE_VAL_TARGETS = "val_targets.npy"

    # =========================================
    # Model Architecture (HC-ZI-ResDnCNN)
    # =========================================
    MODEL_DEPTH = 30  # Number of residual blocks (implies ~60+ conv layers)
    MODEL_FILTERS = 96  # High capacity width
    INPUT_CHANNELS = 1  # Grayscale input
    OUTPUT_CHANNELS = 1  # Predicting residual noise (single channel)
    KERNEL_SIZE = 3  # Standard 3x3 convolution

    # =========================================
    # Training Hyperparameters
    # =========================================
    NUM_ENSEMBLE_MODELS = 5  # Number of independent models to train
    SEED = 42  # Base seed for reproducibility

    # Optimization
    EPOCHS = 50  # Max epochs per model (constrained by runtime)
    BATCH_SIZE = 128  # Tuned for A100 40GB
    LEARNING_RATE = 1e-4  # Initial learning rate for AdamW
    WEIGHT_DECAY = 1e-4  # Regularization
    GRAD_CLIP = 1.0  # Gradient clipping value
    PATIENCE = 8  # Early stopping patience

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================
    # Hardware & Compute
    # =========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # Inference
    TTA_STEPS = 8  # Number of Test Time Augmentations (flips/rotations)
