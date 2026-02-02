import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Idea 5: Cascade R-CNN with ResNeXt-101 and Global Context Modeling.
    """

    # ==============================
    # General Configuration
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==============================
    # Directories and Paths
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Metadata paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "cascade_rcnn_resnext101_best.pth")
    LOG_PATH = os.path.join(WORKING_DIR, "training_log.csv")

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Data & Preprocessing
    # ==============================
    IMAGE_SIZE = 800  # Target size for Letterbox resizing (longest edge)
    NUM_WORKERS = 4  # Number of data loading workers

    # Class Definitions
    # Study Level: Negative, Typical, Indeterminate, Atypical
    STUDY_CLASSES = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    NUM_STUDY_CLASSES = len(STUDY_CLASSES)

    # Image Level: Background (0) + Opacity (1)
    # Note: Faster/Cascade R-CNN usually handles background internally,
    # so num_classes often equals classes + 1 (background) depending on implementation.
    # Here we define the number of object categories.
    NUM_OBJECT_CLASSES = 2  # 0: Background, 1: Opacity

    # ==============================
    # Model Architecture
    # ==============================
    BACKBONE_NAME = "resnext101_32x4d"
    PRETRAINED = True

    # Cascade R-CNN Thresholds
    CASCADE_IOU_THRESHOLDS = [0.5, 0.6, 0.7]

    # ==============================
    # Training Hyperparameters
    # ==============================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Batch size adjusted for A100 GPU (40GB) and ResNeXt-101 memory footprint
    BATCH_SIZE = 8
    NUM_EPOCHS = 10

    # Optimizer (SGD with Momentum is standard for R-CNN)
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001

    # Scheduler (Linear Warmup + Step Decay)
    WARMUP_EPOCHS = 1
    LR_STEP_SIZE = 3
    LR_GAMMA = 0.1

    # Loss Weights
    LOSS_WEIGHT_RPN_CLS = 1.0
    LOSS_WEIGHT_RPN_BOX = 1.0
    LOSS_WEIGHT_ROI_CLS = 1.0
    LOSS_WEIGHT_ROI_BOX = 1.0
    LOSS_WEIGHT_STUDY = 1.0  # Weight for the auxiliary study classification head

    # ==============================
    # Inference & Post-processing
    # ==============================
    # Test Time Augmentation
    USE_TTA = True  # Use Horizontal Flip TTA

    # Weighted Boxes Fusion (WBF)
    WBF_IOU_THRESHOLD = 0.5
    WBF_CONF_THRESHOLD = 0.01  # Skip boxes with very low confidence before fusion

    # Final Prediction Formatting
    CONFIDENCE_THRESHOLD_SUBMISSION = (
        0.001  # Threshold for including in submission string
    )

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets the random seeds for python, numpy, and torch to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms can slow down training, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
