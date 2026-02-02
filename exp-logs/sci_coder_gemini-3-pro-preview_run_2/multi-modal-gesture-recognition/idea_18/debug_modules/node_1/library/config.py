import os
import torch


class Config:
    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific cache directory for this idea
    CACHE_DIR = "./working/idea_18/"
    SUBMISSION_DIR = "./submission/"

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Preprocessing & Feature Engineering
    # ==========================================
    # Skeleton
    # Using 12 Upper-Body Joints:
    # HipCenter(0), Spine(1), ShoulderCenter(2), Head(3),
    # ShoulderLeft(4), ElbowLeft(5), WristLeft(6), HandLeft(7),
    # ShoulderRight(8), ElbowRight(9), WristRight(10), HandRight(11)
    JOINTS_TO_USE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(JOINTS_TO_USE)
    COORD_DIM = 3  # X, Y, Z

    # Normalization
    SCALE_FACTOR = 0.001  # Convert mm to meters
    ROOT_JOINT_IDX = 0  # HipCenter for centering

    # Audio
    AUDIO_SR = 16000
    AUDIO_N_MFCC = 13
    AUDIO_N_FFT = 2048
    AUDIO_HOP_LENGTH = 512

    # Feature Dimensions
    # Input: (Joints * Coords) + (Joints * Velocity) + Audio
    # (12 * 3) + (12 * 3) + 13 = 36 + 36 + 13 = 85
    INPUT_DIM = (NUM_JOINTS * COORD_DIM * 2) + AUDIO_N_MFCC

    # ==========================================
    # Label Configuration
    # ==========================================
    # 20 Gestures + 1 Background
    NUM_CLASSES = 21
    BACKGROUND_CLASS_IDX = 0

    # Gesture Vocabulary (for reference/decoding)
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

    # ==========================================
    # Model Architecture (DSL-CRCN)
    # ==========================================
    # Stage 1: Bi-LSTM
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True

    # Stage 2 & 3: Dual-Scale TCN
    TCN_CHANNELS = 256
    TCN_NUM_LAYERS = 10
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # Latent Transition Head
    TRANSITION_CHANNELS = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Training Loop Control
    MAX_EPOCHS = 70
    EARLY_STOPPING_PATIENCE = 10

    # Debugging: Set to an integer (e.g., 50) to limit dataset size, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Loss Function Configuration
    # ==========================================
    # Class Weights: 0.1 for Background, 1.0 for Gestures
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # Loss Component Weights
    # Total = L_stage1 + L_stage2 + L_stage3 + Smoothness
    # Smoothness (T-MSE) weight
    TMSE_WEIGHT = 0.15

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_class_weights_tensor():
        return torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(Config.DEVICE)
