import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True
    NUM_WORKERS = 4  # Number of DataLoader workers

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Input Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    ANNOTATION_PATH = os.path.join(INPUT_DIR, "train_annotations.csv")

    # Output Paths
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Labels
    LABEL_COLS = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]
    NUM_CLASSES = len(LABEL_COLS)

    # Image Processing
    IMAGE_SIZE = (640, 640)  # (Height, Width)
    IN_CHANNELS = 3

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    BACKBONE = "tf_efficientnetv2_s"
    PRETRAINED = True
    DROP_RATE = 0.2
    DROP_PATH_RATE = 0.1

    # Multi-Scale Feature Aggregation
    # Strides to extract features from the backbone for aggregation
    FEATURE_SCALES = [8, 16, 32]

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 10

    # Batch Size Strategy
    # Use Gradient Accumulation to reach effective batch size >= 32
    BATCH_SIZE = 8  # Physical batch size per step (fits in GPU memory)
    EFFECTIVE_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = max(1, EFFECTIVE_BATCH_SIZE // BATCH_SIZE)

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    MIN_LR = 1e-6

    # Loss Weights
    CLS_LOSS_WEIGHT = 1.0
    AUX_LOSS_WEIGHT = 1.0  # Weight for the auxiliary segmentation head (Pixel-wise BCE)

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility across random, numpy, and torch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
