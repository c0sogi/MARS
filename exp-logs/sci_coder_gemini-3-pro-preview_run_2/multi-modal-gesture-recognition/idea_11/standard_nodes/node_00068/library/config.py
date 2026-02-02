import os
import torch


class Config:
    """
    Configuration class for the NMD-CRCN model pipeline.
    Centralizes all hyperparameters, file paths, and constants.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Main working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_11"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure all working directories exist immediately
    for _dir in [WORKING_DIR, CACHE_DIR, CHECKPOINTS_DIR, SUBMISSION_DIR]:
        os.makedirs(_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Gesture Vocabulary Mapping (Name -> ID)
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

    # Reverse Mapping (ID -> Name)
    ID_TO_NAME = {v: k for k, v in GESTURE_MAP.items()}

    # Total Classes: Background (0) + 20 Gestures
    NUM_CLASSES = 21

    # Skeleton Configuration
    # We strictly use the 12 Upper-Body Joints.
    # Based on the dataset description order:
    # 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
    # 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
    # 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
    SELECTED_JOINTS_INDICES = list(range(12))
    NUM_JOINTS = 12

    # Audio Configuration
    AUDIO_MFCC_N_MFCC = 13

    # Input Feature Dimension Construction
    # Features = [Joint Positions (x,y,z)] + [Joint Velocities (dx,dy,dz)] + [Audio MFCCs]
    # Dim = (12 * 3) + (12 * 3) + 13 = 36 + 36 + 13 = 85
    INPUT_DIM = (NUM_JOINTS * 3) + (NUM_JOINTS * 3) + AUDIO_MFCC_N_MFCC

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Stage 1: Recurrent Encoder (Bi-LSTM)
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3  # Dropout between LSTM layers

    # Stage 2 & 3: Deep TCN (Refinement)
    TCN_NUM_LAYERS = 10
    TCN_NUM_CHANNELS = 128
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 8
    NUM_EPOCHS = 60
    PATIENCE = 15  # Early stopping patience

    # Class Weights for Loss Function
    # Aggressive weighting: 0.1 for Background (Index 0), 1.0 for all Gestures (Indices 1-20)
    CLASS_WEIGHTS = [0.1] + [1.0] * 20
