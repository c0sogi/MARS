import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for the specific idea
    WORKING_DIR = "./working/idea_7"

    # Sub-directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data & Feature Extraction
    # =========================================================================
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

    # Class Definitions
    # 0 is Background, 1-20 are Gestures
    NUM_CLASSES = 21

    # Skeleton Features
    # We select only the 12 Upper-Body Joints based on the dataset description order
    # Indices: 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
    #          4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
    #          8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINTS)

    # Input Dimensions
    # Per joint: (x, y, z) normalized + (dx, dy, dz) velocity = 6 features
    # Total Skeleton Features: 12 joints * 6 = 72
    SKELETON_INPUT_SIZE = NUM_JOINTS * 6

    # Audio Features
    AUDIO_N_MFCC = 13
    AUDIO_INPUT_SIZE = AUDIO_N_MFCC

    # Total Input Dimension for the Model
    INPUT_DIM = SKELETON_INPUT_SIZE + AUDIO_INPUT_SIZE

    # =========================================================================
    # Model Architecture (IDC-RCN)
    # =========================================================================
    # Stage 1: Recurrent Encoder
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True

    # Stage 2 & 3: Temporal Convolutional Networks (Coarse & Fine Refinement)
    TCN_NUM_CHANNELS = [256] * 10  # 10 layers, keeping channel depth constant
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = (
        4  # Small batch size due to variable sequence lengths and high memory usage
    )
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 10

    # Loss Function Configuration
    # Class Weights: 0.1 for Background (index 0), 1.0 for Gestures (indices 1-20)
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # Truncated Mean Squared Error (T-MSE) for Smoothing
    # Applied to Stage 2 and Stage 3 probabilities
    TMSE_THRESHOLD = 4.0  # Standard threshold for T-MSE
    LAMBDA_TMSE = 3.0  # Increased weight for smoothing (Cite Lesson 00031)

    # =========================================================================
    # Inference & Post-Processing
    # =========================================================================
    MEDIAN_FILTER_KERNEL = 7  # Size of the median filter window

    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def ensure_dirs(cls):
        """Creates necessary directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.ensure_dirs()
