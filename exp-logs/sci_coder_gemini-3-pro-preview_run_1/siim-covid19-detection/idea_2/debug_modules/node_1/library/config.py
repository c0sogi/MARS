import os
import torch


class Config:
    """
    Configuration class for the COVID-19 Radiography Classification and Detection Task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # ====================================================
    # General & Reproducibility
    # ====================================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Limits dataset size when DEBUG is True

    # ====================================================
    # Compute Environment
    # ====================================================
    # A100 GPU allows for batch size 16 with EfficientNet-B3 at 512x512
    BATCH_SIZE = 16
    NUM_WORKERS = 12  # Matches available vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Data Directories & Paths
    # ====================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ====================================================
    # Working Directory & Caching
    # ====================================================
    # Directory for intermediate files, checkpoints, and cache
    WORKING_DIR = "./working/idea_2"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Paths (Numpy/Parquet) for deterministic data loading
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
    CACHE_TRAIN_MASKS = os.path.join(WORKING_DIR, "train_masks.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
    CACHE_VAL_MASKS = os.path.join(WORKING_DIR, "val_masks.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
    CACHE_TEST_DIMS = os.path.join(WORKING_DIR, "test_dims.parquet")

    # ====================================================
    # Submission
    # ====================================================
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    BACKBONE = "efficientnet-b3"
    ENCODER_WEIGHTS = "imagenet"

    # Study Level Classes: Negative, Typical, Indeterminate, Atypical
    NUM_STUDY_CLASSES = 4

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    IMG_SIZE = (512, 512)
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Hybrid Loss Weights
    # Segmentation Loss = BCE * 0.5 + Dice * 0.5
    SEG_LOSS_BCE_WEIGHT = 0.5
    SEG_LOSS_DICE_WEIGHT = 0.5

    # Total Loss = Class_Loss * 1.0 + Seg_Loss * 1.0
    TOTAL_LOSS_CLASS_WEIGHT = 1.0
    TOTAL_LOSS_SEG_WEIGHT = 1.0

    # Optimization Strategy
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    EARLY_STOPPING_PATIENCE = 5

    # ====================================================
    # Augmentation (Albumentations)
    # ====================================================
    # CoarseDropout settings: ~10% of image size (50px)
    COARSE_DROPOUT_PARAMS = {
        "max_holes": 8,
        "max_height": 52,
        "max_width": 52,
        "min_holes": 1,
        "min_height": 20,
        "min_width": 20,
        "fill_value": 0,
        "p": 0.5,
    }

    # ====================================================
    # Inference Thresholds
    # ====================================================
    PIXEL_THRESHOLD = 0.5  # For generating binary mask from heatmap
    CONFIDENCE_THRESHOLD = 0.001  # Minimum confidence to keep a bounding box
