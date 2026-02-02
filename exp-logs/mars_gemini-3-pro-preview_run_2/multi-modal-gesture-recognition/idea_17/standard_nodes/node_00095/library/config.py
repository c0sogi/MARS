import os
import torch


class Config:
    """
    Configuration for the Supervised Gated-Cascaded Recurrent-Convolutional Network (SG-CRCN).
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Cache File Paths
    TRAIN_CACHE_FILE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE_FILE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE_FILE = os.path.join(WORKING_DIR, "test_data.npz")

    # Checkpoint Path
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Path
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
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

    # Inverse mapping for decoding
    ID_TO_NAME = {v: k for k, v in GESTURE_MAP.items()}

    # Skeleton Features
    # Indices for 12 Upper-Body Joints based on the dataset description list:
    # 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
    # 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
    # 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_SELECTED_JOINTS = len(SELECTED_JOINTS)

    # Normalization: Scale mm to meters
    SCALE_FACTOR = 0.001

    # Audio Features
    AUDIO_MFCC_N_MFCC = 13

    # Input Dimensionality Calculation
    # Skeleton Position: 12 joints * 3 (x, y, z)
    # Skeleton Velocity: 12 joints * 3 (vx, vy, vz)
    # Audio: 13 MFCCs
    INPUT_DIM = (
        (NUM_SELECTED_JOINTS * 3) + (NUM_SELECTED_JOINTS * 3) + AUDIO_MFCC_N_MFCC
    )

    # -------------------------------------------------------------------------
    # Model Architecture (SG-CRCN)
    # -------------------------------------------------------------------------
    HIDDEN_DIM = 256
    NUM_CLASSES = 21  # 0 (Background) + 20 Gestures

    # Stage 1: Bi-LSTM
    LSTM_LAYERS = 2

    # Stages 2 & 3: Gated MS-TCN
    # Dilations: 2^0, 2^1, ..., 2^9 (1 to 512)
    NUM_TCN_LAYERS = 10
    KERNEL_SIZE = 3
    DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8
    NUM_EPOCHS = 60  # Sufficient for convergence with early stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0

    # Early Stopping
    PATIENCE = 12

    # -------------------------------------------------------------------------
    # Loss Configuration
    # -------------------------------------------------------------------------
    # Class Weights: 0.1 for Background, 1.0 for Gestures
    # We create a tensor here, but it should be moved to device during training
    CLASS_WEIGHTS_INIT = [0.1] + [1.0] * 20

    # Multi-Task Loss Weights
    W_CLS = 1.0  # Classification Loss
    W_BND = 1.0  # Boundary Loss (Crucial for Gated Units)
    W_SMOOTH = 0.15  # Unclamped T-MSE Smoothing Loss

    # -------------------------------------------------------------------------
    # Inference / Post-Processing
    # -------------------------------------------------------------------------
    MEDIAN_WINDOW = 7  # Window size for median filtering
