import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Definitions
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

    # Inverse map for decoding
    ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

    # Number of classes: 20 gestures + 1 background (0)
    NUM_CLASSES = 21

    # Skeleton Configuration
    # We select the 12 Upper-Body Joints based on the dataset description order:
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head,
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft,
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINTS)

    # Input Feature Dimensions
    # Skeleton: 3 coords (x,y,z) * 12 joints = 36
    # Velocity: 3 coords (vx,vy,vz) * 12 joints = 36
    # Audio: 13 MFCCs
    # Total Input Dim = 36 + 36 + 13 = 85
    INPUT_DIM = (NUM_JOINTS * 3) + (NUM_JOINTS * 3) + 13

    # -------------------------------------------------------------------------
    # Audio Parameters
    # -------------------------------------------------------------------------
    AUDIO_SAMPLE_RATE = 16000  # Based on dataset analysis
    N_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512  # Defines temporal resolution of audio features

    # -------------------------------------------------------------------------
    # Model Architecture Parameters
    # -------------------------------------------------------------------------
    # Stage 1: LSTM
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3

    # Stage 2 & 3: TCN
    TCN_NUM_CHANNELS = [64] * 10  # 10 layers (Cite Lesson 00063)
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3
    # Dilations: 1, 2, 4, 8, ..., 512
    TCN_DILATIONS = [2**i for i in range(10)]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50  # Can be adjusted
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights
    # Class weights: Background (0) vs Gestures (1-20)
    # Background gets 0.1 to prevent it from dominating the loss
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[0] = 0.1

    # Boundary Loss
    BOUNDARY_POS_WEIGHT = 10.0  # Weight for positive boundary samples

    # Multi-Task Loss Components
    LOSS_LAMBDA_CLS = 1.0  # Classification Loss
    LOSS_LAMBDA_BND = 0.0  # Boundary Loss (Cite Lesson 00077)
    LOSS_LAMBDA_SMOOTH = 1.0  # Probability Smoothing Loss (T-MSE) (Cite Lesson 00077)

    # -------------------------------------------------------------------------
    # Augmentation Parameters (Physically Consistent Smooth Noise)
    # -------------------------------------------------------------------------
    AUG_SIGMA = 0.01  # Standard deviation of Gaussian noise
    AUG_SMOOTH_KERNEL = 5  # Kernel size for temporal low-pass filter

    # -------------------------------------------------------------------------
    # Inference / Post-Processing
    # -------------------------------------------------------------------------
    MEDIAN_FILTER_KERNEL = 7  # Size of median filter for label smoothing
