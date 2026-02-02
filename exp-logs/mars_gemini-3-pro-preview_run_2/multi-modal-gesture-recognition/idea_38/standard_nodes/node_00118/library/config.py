import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_38"

    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    for d in [WORKING_DIR, CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Gesture Vocabulary
    GESTURE_MAP = {
        "vattene": 1,
        "vieniqui": 2,
        "perfetto": 3,
        "furbo": 4,
        "cheduepalle": 5,
        "chevuoi": 6,
        "daccordo": 7,
        "seipazzo": 8,
        "combinato": 9,
        "freganiente": 10,
        "ok": 11,
        "cosatifarei": 12,
        "basta": 13,
        "prendere": 14,
        "noncenepiu": 15,
        "fame": 16,
        "tantotempo": 17,
        "buonissimo": 18,
        "messidaccordo": 19,
        "sonostufo": 20,
    }
    # 0 is reserved for background/null
    NUM_CLASSES = 21

    # Feature Engineering
    # 12 Upper Body Joints
    UPPER_BODY_JOINTS = [
        "HipCenter",
        "Spine",
        "ShoulderCenter",
        "Head",
        "ShoulderLeft",
        "ElbowLeft",
        "WristLeft",
        "HandLeft",
        "ShoulderRight",
        "ElbowRight",
        "WristRight",
        "HandRight",
    ]
    NUM_JOINTS = 12
    COORDS_PER_JOINT = 3

    # Dimensions
    SKELETON_DIM = NUM_JOINTS * COORDS_PER_JOINT  # 36
    VELOCITY_DIM = NUM_JOINTS * COORDS_PER_JOINT  # 36
    AUDIO_MFCC_DIM = 13

    # Total Input Dimension: Position + Velocity + Audio
    INPUT_DIM = SKELETON_DIM + VELOCITY_DIM + AUDIO_MFCC_DIM  # 85

    # ==========================================
    # Model Architecture (HCRG-CN)
    # ==========================================
    HIDDEN_DIM = 256
    LSTM_LAYERS = 2
    KERNEL_SIZE_STEM = 3
    NUM_STAGES = 3  # Encoder + 2 Refinement Stages

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Weights
    # Classification: Background (0.1) vs Gestures (1.0)
    CLASS_WEIGHTS_TENSOR = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS_TENSOR[0] = 0.1

    # Multi-Task Weights
    W_CLS = 1.0  # Classification
    W_BND = 0.5  # Boundary (Focal)
    W_SMOOTH = 0.15  # Smoothing (T-MSE)

    # Focal Loss Parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # T-MSE Parameters
    TMSE_THRESHOLD = 4.0

    # ==========================================
    # Augmentation & Preprocessing
    # ==========================================
    SCALE_FACTOR = 0.001  # Convert mm to meters
    AUG_NOISE_SIGMA = 0.001
    AUG_FILTER_SIZE = 3

    # ==========================================
    # Inference
    # ==========================================
    MEDIAN_FILTER_KERNEL = 7
