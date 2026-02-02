import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Ensure working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

    # Submission output
    SUBMISSION_PATH = "./submission.csv"

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Image resizing for training and inference
    # High resolution is critical for small kuzushiji characters
    MIN_SIZE = 1024
    MAX_SIZE = 2048

    # Batch size and workers
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "cascade_rcnn_resnet50_fpn"

    # 3848 Unicode characters + 1 Background class
    NUM_CLASSES = 3849

    # =========================================================================
    # Training Configuration
    # =========================================================================
    # Optimizer settings (Linear Scaling Rule: 0.02 * 4/16 -> ~0.005)
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    # Scheduler settings
    EPOCHS = 15
    LR_STEPS = [10, 13]
    LR_GAMMA = 0.1

    # Gradient clipping (optional but recommended for stability)
    GRADIENT_CLIP = 10.0

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Confidence threshold to balance Precision/Recall
    SCORE_THRESH = 0.35

    # Maximum number of predictions per image (Task limit is 1200)
    DETECTIONS_PER_IMG = 1200

    # RPN Settings for Inference
    # Keep more proposals to handle dense pages (~600 chars)
    RPN_POST_NMS_TOP_N_TEST = 2000

    # NMS Threshold for RPN and ROI heads
    NMS_THRESH = 0.5

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the seed for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior on CUDA (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
