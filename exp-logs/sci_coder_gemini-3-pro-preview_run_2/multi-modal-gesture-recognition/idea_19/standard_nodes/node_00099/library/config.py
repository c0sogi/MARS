import os


class Config:
    # -------------------------------------------------------------------------
    # Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data (Idea 19 specific)
    CACHE_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
    # Scale factor to convert millimeters to meters (Crucial for stability of Gated Units)
    SCALE_FACTOR = 0.001

    # Skeleton Joints: 12 Upper-Body Joints
    # Indices based on Kinect format provided in description:
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    SKELETON_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SKELETON_JOINTS)
    CHANNELS_PER_JOINT = 3  # X, Y, Z

    # Audio Features
    USE_AUDIO = True
    AUDIO_MFCC_N_MFCC = 13

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
    # Class 0 is Background, 1-20 are gestures
    NUM_CLASSES = 21

    # -------------------------------------------------------------------------
    # Model Architecture (RSG-CRCN)
    # -------------------------------------------------------------------------
    # Stage 1: Recurrent Encoder
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True

    # Stage 2 & 3: Gated MS-TCN
    TCN_NUM_LAYERS = 10
    TCN_CHANNELS = 256
    TCN_KERNEL_SIZE = 3
    # Dilations: 2^0 to 2^9 (1, 2, 4, ..., 512)
    TCN_DILATIONS = [2**i for i in range(TCN_NUM_LAYERS)]

    DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 4  # Small batch size due to variable sequence lengths
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # Loss Weights
    # Class weights: 0.1 for Background (0), 1.0 for Gestures (1-20)
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # Multi-task Loss Components
    LAMBDA_CLS = 1.0  # Classification Loss
    LAMBDA_BND = 1.0  # Boundary Loss (Explicit supervision)
    LAMBDA_SMOOTH = 0.15  # T-MSE Smoothing Loss (Unclamped)

    # -------------------------------------------------------------------------
    # Debugging
    # -------------------------------------------------------------------------
    # Set to a small integer (e.g., 50) to train on a subset for debugging
    # Set to None to train on the full dataset
    DEBUG_SUBSET_SIZE = None
