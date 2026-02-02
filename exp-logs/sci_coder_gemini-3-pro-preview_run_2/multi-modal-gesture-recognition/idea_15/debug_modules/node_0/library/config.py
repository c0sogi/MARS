import os
import torch


class Config:
    """
    Configuration class for the Supervised Gated-Cascaded Recurrent-Convolutional Network (SG-CRCN).
    Handles paths, hyperparameters, and constants for data processing, training, and inference.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
    SEED = 42

    # Gesture Vocabulary (Name to ID mapping)
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

    # Total classes: 20 gestures + 1 background class (index 0)
    NUM_CLASSES = 21

    # Skeleton Configuration
    # Selecting 12 Upper-Body Joints based on dataset description
    # Indices: 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
    # 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
    # 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINTS)

    # Normalization
    SCALE_FACTOR = 0.001  # Convert millimeters to meters
    CENTER_JOINT_IDX = 0  # HipCenter for centering coordinates

    # Audio Configuration
    AUDIO_SR = 16000
    N_MFCC = 13

    # Input Feature Dimension
    # Per frame: (Joints * 3 coords) + (Joints * 3 velocity) + Audio MFCCs
    INPUT_DIM = (NUM_JOINTS * 3) + (NUM_JOINTS * 3) + N_MFCC

    # -------------------------------------------------------------------------
    # Model Architecture (SG-CRCN)
    # -------------------------------------------------------------------------
    # Stage 1: Multi-Task Recurrent Encoder (Bi-LSTM)
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3

    # Stage 2 & 3: Supervised Gated Refinement (Gated MS-TCN)
    MSTCN_STAGES = 2  # Number of refinement stages after the encoder
    MSTCN_LAYERS = 10  # Layers per stage (dilations 2^0 to 2^9)
    MSTCN_CHANNELS = 256
    MSTCN_KERNEL_SIZE = 3
    MSTCN_DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # Loss Component Weights
    CLS_LOSS_WEIGHT = 1.0  # Classification Loss
    BND_LOSS_WEIGHT = 1.0  # Boundary Regression Loss
    SMOOTH_LOSS_WEIGHT = 0.15  # Truncated MSE (Smoothing) Loss

    # Class Imbalance Handling
    # Background class (0) is frequent, so we downweight it
    BG_WEIGHT = 0.1

    # -------------------------------------------------------------------------
    # Inference & Post-Processing
    # -------------------------------------------------------------------------
    MEDIAN_FILTER_KERNEL = 15  # Kernel size for post-prediction smoothing

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use when DEBUG is True

    @staticmethod
    def get_class_weights(device):
        """
        Returns the class weights tensor for CrossEntropyLoss.
        Background class (0) gets lower weight to focus learning on gestures.
        """
        weights = torch.ones(Config.NUM_CLASSES, device=device)
        weights[0] = Config.BG_WEIGHT
        return weights

    @staticmethod
    def get_device():
        """Returns the appropriate torch device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
