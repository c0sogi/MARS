import os
import torch


class Config:
    """
    Configuration class for the Salt Segmentation task using
    FP32-Stabilized Marginalized-Distillation to Multi-Task Student strategy.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = "./working/idea_42"
    SUBMISSION_DIR = "./submission"

    # Ensure working and cache directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV_PATH = os.path.join(INPUT_DIR, "depths.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    ORIG_SIZE = 101  # Original image size
    IMG_SIZE = 128  # Input size for model (padded/resized)
    CHANNELS = 1  # Grayscale input
    NUM_CLASSES = 1  # Binary segmentation

    # Normalization (ImageNet stats for 1 channel or custom)
    # Using standard ImageNet mean/std for consistency with pretrained weights
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # -------------------------------------------------------------------------
    # Model Parameters
    # -------------------------------------------------------------------------
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for 128x128 on A100
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Stabilization Strategy
    USE_AMP = False  # Strictly FP32 to prevent instability with Lovasz/Ranking losses

    # Epochs
    EPOCHS_STAGE1 = 50  # Specialist Teacher (Supervised)
    EPOCHS_STAGE3 = 50  # Generalist Student (Distillation)

    # Loss Weights (Stage 3 Student)
    LOSS_WEIGHT_BCE = 1.0  # For soft targets (unlabeled) and hard targets (labeled)
    LOSS_WEIGHT_LOVASZ = 1.0  # For labeled data only
    LOSS_WEIGHT_MSE = 1.0  # For auxiliary depth head

    # -------------------------------------------------------------------------
    # Augmentation Parameters
    # -------------------------------------------------------------------------
    # Non-Rigid (Elastic) - Critical for this dataset
    AUG_ELASTIC_ALPHA = 120
    AUG_ELASTIC_SIGMA = 6
    AUG_ELASTIC_P = 0.2

    # Rigid (Geometric)
    AUG_RIGID_P = 0.2

    # -------------------------------------------------------------------------
    # Marginalization Strategy (Stage 2)
    # -------------------------------------------------------------------------
    # Depth values (in Standard Deviations) to scan for generating robust pseudo-labels
    DEPTH_SCAN_SIGMAS = [-1.5, -0.75, 0.0, 0.75, 1.5]

    # -------------------------------------------------------------------------
    # Post-Processing & Evaluation
    # -------------------------------------------------------------------------
    # Gating: Minimum mAP required to include a Teacher fold in the ensemble
    TEACHER_GATING_MAP = 0.75

    # Threshold Optimization Range
    THRESH_START = 0.5
    THRESH_END = 0.95
    THRESH_STEP = 0.05

    # Test Time Augmentation
    TTA_FLIP = True

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set to an integer (e.g., 100) to limit dataset size for rapid testing.
    # Set to None for full training run.
    DEBUG_SAMPLE_SIZE = None
