import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the Multi-Task Faster R-CNN with Spatial Attention.
    Defines hyperparameters for data, model, training, and inference.
    """

    # ==============================
    # Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Ensure working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_DIR = "./demo/submission"
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==============================
    # Data Configuration
    # ==============================
    IMAGE_SIZE = 800  # Fixed resolution as per strategy
    BATCH_SIZE = 8  # Tuned for A100 40GB
    NUM_WORKERS = 4  # Parallel data loading

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # ==============================
    # Model Architecture
    # ==============================
    BACKBONE = "resnet101"  # ResNet-101-FPN

    # Detection Head Classes
    # 0: Background
    # 1: Typical Appearance
    # 2: Indeterminate Appearance
    # 3: Atypical Appearance
    NUM_DETECTION_CLASSES = 4

    # Study Head Classes (for Spatial Attention)
    # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
    NUM_STUDY_CLASSES = 4

    # RPN Configuration (High Capacity Settings)
    # Increased to overcome density bottleneck
    RPN_PRE_NMS_TOP_N_TRAIN = 4000
    RPN_POST_NMS_TOP_N_TRAIN = 3000
    RPN_PRE_NMS_TOP_N_TEST = 2000
    RPN_POST_NMS_TOP_N_TEST = 1000

    # ROI Heads Configuration
    DETECTIONS_PER_IMG = 200  # Allowed detections per image
    ROI_HEADS_SCORE_THRESH = 0.05  # Low threshold to capture weak signals
    ROI_HEADS_NMS_THRESH = 0.5  # Standard NMS

    # ==============================
    # Training Hyperparameters
    # ==============================
    NUM_EPOCHS = 12

    # Optimizer
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9

    # Scheduler
    # Decay LR after 60% of total epochs (approx epoch 7-8)
    LR_DECAY_STEP = int(NUM_EPOCHS * 0.6)
    LR_GAMMA = 0.1
    WARMUP_EPOCHS = 1  # Linear warmup for stability

    # Optimization Stability
    GRAD_CLIP = 10.0  # Gradient clipping norm
    AUX_LOSS_WEIGHT = 0.5  # Weight for Spatial Attention Head loss

    # ==============================
    # Reproducibility
    # ==============================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================
    # Label Mappings
    # ==============================
    # Map study labels to integer IDs
    STUDY_CLASS_MAP = {
        "Negative for Pneumonia": 0,
        "Typical Appearance": 1,
        "Indeterminate Appearance": 2,
        "Atypical Appearance": 3,
    }

    # Inverse mapping for submission generation
    ID_TO_STUDY_CLASS = {v: k for k, v in STUDY_CLASS_MAP.items()}

    # Map detection class IDs to submission strings
    # Note: We collapse 1, 2, 3 to 'opacity' during inference
    DETECTION_CLASS_MAP = {1: "opacity", 2: "opacity", 3: "opacity"}


def setup_reproducibility(seed=Config.SEED):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure
    reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
