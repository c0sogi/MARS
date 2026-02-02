import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Idea 13: Multi-Task DINO with RoI-Transformer Diagnosis Module.
    """

    # ==========================
    # General Setup
    # ==========================
    SEED = 42
    EXPERIMENT_NAME = "idea_13"
    DEBUG = False  # Set to True for fast debugging runs

    # ==========================
    # Directories
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{EXPERIMENT_NAME}"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # ==========================
    # File Paths
    # ==========================
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================
    # Data Preprocessing
    # ==========================
    IMG_SIZE = 1024  # Target size for letterbox resizing
    NUM_WORKERS = 4

    # ==========================
    # Model Architecture
    # ==========================
    # Backbone
    BACKBONE = "swin_large_patch4_window12_384"

    # DINO Detector
    NUM_QUERIES = 300
    NUM_CLASSES_DETECTION = 1  # "opacity"

    # Study Classifier (RoI-Transformer)
    NUM_CLASSES_STUDY = 4  # Negative, Typical, Indeterminate, Atypical
    ROI_ALIGN_SIZE = (7, 7)
    ROI_HEAD_DIM = 256
    ROI_NUM_HEADS = 8
    ROI_DROPOUT = 0.1

    # ==========================
    # Training Hyperparameters
    # ==========================
    # Swin-L at 1024 is memory intensive.
    # A100 40GB can likely handle batch size 2-4 depending on gradient checkpointing.
    BATCH_SIZE = 2
    ACCUMULATE_GRAD_BATCHES = 4  # Effective batch size = 8

    EPOCHS = 15
    LEARNING_RATE = 1e-4
    BACKBONE_LR = 1e-5  # Lower LR for pretrained backbone
    WEIGHT_DECAY = 1e-4
    CLIP_MAX_NORM = 0.1

    # Loss Coefficients
    LOSS_COEF_BOX = 5.0
    LOSS_COEF_GIOU = 2.0
    LOSS_COEF_CLASS = 1.0
    LOSS_COEF_STUDY = 2.0  # Weight for the study-level classification

    # Early Stopping
    PATIENCE = 4

    # ==========================
    # Inference
    # ==========================
    CONFIDENCE_THRESHOLD = 0.001  # Keep low for mAP calculation

    # ==========================
    # Hardware
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the experiment environment:
        1. Create working and cache directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")
