import os
import torch


class Config:
    """
    Configuration for the Dual-Scale Supervised Gated-Cascaded Recurrent-Convolutional Network (DSG-CRCN).
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Gesture Vocabulary
    # -------------------------------------------------------------------------
    # Mapping from gesture name to ID (1-20)
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
    INV_GESTURE_MAP = {v: k for k, v in GESTURE_MAP.items()}

    # Total classes: 20 gestures + 1 background (Index 0)
    NUM_CLASSES = 21

    # -------------------------------------------------------------------------
    # Input Feature Configuration
    # -------------------------------------------------------------------------
    # Skeleton Data
    # We select only the upper body joints to focus on the gesture semantics
    # and reduce noise from lower body movements.
    # Indices based on the provided Kinect structure:
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    HIP_CENTER_INDEX = 0  # Used for centering normalization

    NUM_JOINTS = len(UPPER_BODY_INDICES)
    INPUT_DIM_SKELETON = NUM_JOINTS * 3  # (x, y, z) coordinates

    # Audio Data
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13
    INPUT_DIM_AUDIO = N_MFCC

    # Combined Input Dimension
    # The model receives: [Skeleton Position (36) + Skeleton Velocity (36) + Audio MFCC (13)]
    # Velocity has the same dimension as Position.
    INPUT_DIM = (INPUT_DIM_SKELETON * 2) + INPUT_DIM_AUDIO

    # Preprocessing Constants
    SCALE_FACTOR = 0.001  # Convert millimeters to meters

    # -------------------------------------------------------------------------
    # Model Architecture (DSG-CRCN)
    # -------------------------------------------------------------------------
    # Stage 1: Bi-LSTM Encoder
    LSTM_LAYERS = 2
    HIDDEN_SIZE = 256

    # Stage 2 & 3: Dual-Scale Gated Refinement
    # Each block has two branches (Global Dilated + Local Fixed)
    BRANCH_CHANNELS = 128  # Channels per branch
    # The branches are concatenated, so the internal block width is BRANCH_CHANNELS * 2 = 256

    NUM_STAGES = 3
    STAGE_LAYERS = 10  # Number of layers in the TCN stages
    KERNEL_SIZE = 3
    DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 8  # Small batch size due to variable length video sequences
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Configuration
    # We strictly downweight the background class to address class imbalance
    BG_WEIGHT = 0.1
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[0] = BG_WEIGHT

    # Multi-Task Loss Weights
    BOUNDARY_LOSS_WEIGHT = 1.0

    # Smoothing Loss (Truncated MSE on Softmax Probabilities)
    TMSE_THRESHOLD = 0.15  # The truncation threshold (tau)
    TMSE_WEIGHT = 0.15  # Weight for the smoothing term in the total loss

    # -------------------------------------------------------------------------
    # Data Augmentation
    # -------------------------------------------------------------------------
    # Physically consistent noise injection
    NOISE_STD = 0.01  # Standard deviation in meters
    TEMPORAL_SMOOTHING_WINDOW = 5  # Window for smoothing generated noise

    # -------------------------------------------------------------------------
    # Inference / Post-Processing
    # -------------------------------------------------------------------------
    MEDIAN_FILTER_KERNEL = 7  # Size of the median filter window for label smoothing

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for quick checks
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50
