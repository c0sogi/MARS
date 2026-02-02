import os
import torch


class Config:
    """
    Global configuration for the GHG-CRCN model pipeline.
    Includes paths, data processing constants, model hyperparameters,
    and training settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and checkpoints
    # Specifically for Idea 27 (GHG-CRCN)
    WORKING_DIR = "./working/idea_27"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Settings
    # =========================================================================
    # Map gesture names to IDs (1-20)
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

    # Skeleton Joint Indices (Kinect Format)
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    # 12-19: Lower body (ignored)
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINTS)

    # Audio Features
    N_MFCC = 13

    # Input Dimension Calculation
    # 1. Joint Positions (x, y, z): 12 joints * 3 = 36
    # 2. Joint Velocities (dx, dy, dz): 12 joints * 3 = 36
    # 3. Bone Vectors (dx, dy, dz): 11 bones connecting 12 joints * 3 = 33
    # 4. Audio MFCCs: 13
    # Total Input Dimension = 36 + 36 + 33 + 13 = 118
    INPUT_DIM = (NUM_JOINTS * 3) + (NUM_JOINTS * 3) + ((NUM_JOINTS - 1) * 3) + N_MFCC

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Classes: 0 = Background, 1-20 = Gestures
    NUM_CLASSES = 21

    # Stage 1: Geometric Recurrent Encoder (Bi-LSTM)
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2

    # Stage 2 & 3: Hierarchical Gated Refinement (Gated MS-TCN)
    TCN_CHANNELS = 256
    TCN_KERNEL_SIZE = 3
    TCN_LAYERS = 10  # Dilations: 1, 2, 4, ..., 512
    DROPOUT = 0.3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Batch size (adjust based on memory, sequences can be long)
    BATCH_SIZE = 8

    # Training duration
    NUM_EPOCHS = 60
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights
    # Class Weights: 0.1 for Background (Index 0), 1.0 for Gestures
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[0] = 0.1

    # Multi-Task Loss Component Weights
    # L_total = W_CLS * L_cls + W_BND * L_bnd + W_FG * L_fg + W_SMOOTH * L_smooth
    W_CLS = 1.0
    W_BND = 1.0
    W_FG = 1.0
    W_SMOOTH = 0.5  # T-MSE Smoothing weight

    # =========================================================================
    # Debug / Development
    # =========================================================================
    # Set to True to train on a small subset for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50
