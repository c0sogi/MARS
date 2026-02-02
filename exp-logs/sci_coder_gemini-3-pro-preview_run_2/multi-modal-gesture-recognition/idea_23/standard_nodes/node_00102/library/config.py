import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Symmetric Gated-Cascaded Recurrent-Convolutional Network (SymG-CRCN)
    task. Centralizes all hyperparameters for data processing, model architecture, training,
    and inference.
    """

    # -------------------------------------------------------------------------
    # 1. Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (using .npz for efficient storage)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")

    # Model Checkpoints
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # -------------------------------------------------------------------------
    # 3. Data Processing & Augmentation
    # -------------------------------------------------------------------------
    # Debugging: Set to a small integer (e.g., 50) to limit dataset size for fast checking
    # Set to None for full training
    DEBUG_SUBSET_SIZE = None

    # Skeleton Configuration
    # We select 12 Upper-Body Joints based on the dataset description
    # Indices: 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
    # 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
    # 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Bone Vectors: Defined as pairs of (Parent, Child) indices
    # Used to compute geometric orientation features
    BONE_PAIRS = [
        (0, 1),  # HipCenter -> Spine
        (1, 2),  # Spine -> ShoulderCenter
        (2, 3),  # ShoulderCenter -> Head
        (2, 4),  # ShoulderCenter -> ShoulderLeft
        (4, 5),  # ShoulderLeft -> ElbowLeft
        (5, 6),  # ElbowLeft -> WristLeft
        (6, 7),  # WristLeft -> HandLeft
        (2, 8),  # ShoulderCenter -> ShoulderRight
        (8, 9),  # ShoulderRight -> ElbowRight
        (9, 10),  # ElbowRight -> WristRight
        (10, 11),  # WristRight -> HandRight
    ]

    # Normalization
    CENTER_JOINT_IDX = 0  # HipCenter used as origin

    # Audio Configuration
    AUDIO_SR = 16000
    N_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512  # ~32ms window

    # Target Generation
    # Sigma for Gaussian smoothing of boundary targets (Soft-Boundary Supervision)
    BOUNDARY_SIGMA = 1.5

    # Input Feature Dimension Calculation
    # 12 Joints * 3 (Position) = 36
    # 12 Joints * 3 (Velocity) = 36
    # 11 Bones * 3 (Vector)    = 33
    # 13 MFCCs                 = 13
    # Total                    = 118
    INPUT_DIM = 118

    # -------------------------------------------------------------------------
    # 4. Model Architecture (SymG-CRCN)
    # -------------------------------------------------------------------------
    # Stage 1: Recurrent Encoder
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True

    # Stage 2 & 3: Symmetric Gated TCN
    TCN_CHANNELS = 256
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3

    # Symmetric "Zoom-Out-Zoom-In" Dilation Schedule
    # Expands receptive field to global context (512) then contracts for local fidelity
    SYMMETRIC_DILATIONS = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        256,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1,
    ]

    # Output
    NUM_CLASSES = 21  # 20 Gestures + 1 Background (Class 0)
    BACKGROUND_CLASS_ID = 0

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights
    # Class weights: Downweight background (0.1) vs Gestures (1.0)
    # Note: Convert to tensor in training script
    CLASS_WEIGHTS_LIST = [0.1] + [1.0] * 20

    # Multi-Task Loss Components
    W_CLS = 1.0  # Classification Loss
    W_BND = 1.0  # Boundary Regression Loss
    W_SMOOTH = 0.15  # Temporal Smoothing (T-MSE) Loss

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    # Median filter kernel size for post-processing predictions
    MEDIAN_FILTER_K = 7
