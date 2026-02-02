import os
import torch


class Config:
    """
    Global configuration for the High-Capacity Non-Causal Sawtooth Network (HC-NCSN).
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this idea
    WORKING_DIR = "./working/idea_26"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache directory for deterministic data processing
    CACHE_DIR = WORKING_DIR

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Processing & Sampling
    # ==========================================
    # Sampling Strategy
    WINDOW_SIZE = 64
    STRIDE = 32

    # Feature Dimensions
    NUM_JOINTS = 20
    # Kinematically Consistent Features: Position(3) + Velocity(3) + Acceleration(3)
    CHANNELS_PER_JOINT = 9
    SKELETON_INPUT_SIZE = NUM_JOINTS * CHANNELS_PER_JOINT  # 180

    # Audio Features
    AUDIO_N_MFCC = 13
    AUDIO_INPUT_SIZE = AUDIO_N_MFCC

    # Total Input Dimension (Early Fusion)
    INPUT_DIM = SKELETON_INPUT_SIZE + AUDIO_INPUT_SIZE  # 193

    # Labels
    # 0: Background, 1-20: Gestures
    NUM_CLASSES = 21

    # ==========================================
    # Model Architecture (HC-NCSN)
    # ==========================================
    # Stage 1: High-Capacity Encoder
    GRU_HIDDEN_SIZE = 128  # Per direction (256 total for Bi-GRU)
    GRU_LAYERS = 1

    # Stage 2 & 3: Non-Causal Sawtooth Refinement
    # Repeated Sawtooth Schedule
    SAWTOOTH_DILATIONS = [1, 2, 4, 8, 1, 2, 4, 8]
    REFINE_CHANNELS = 64
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Loss Configuration
    # Weighted Cross-Entropy: 0.2 for background, 1.0 for gestures
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[0] = 0.2

    # Deep Supervision Weights
    LOSS_WEIGHT_STAGE1 = 1.0
    LOSS_WEIGHT_STAGE2 = 1.0
    LOSS_WEIGHT_STAGE3 = 1.0

    # Log-Space Smoothing Loss
    SMOOTHING_LAMBDA = 0.15
    SMOOTHING_THRESHOLD = 1.0  # Truncated MSE threshold

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    INFERENCE_STRIDE = WINDOW_SIZE // 2  # 50% overlap
    MIN_GESTURE_DURATION = 5  # Minimum frames to keep a gesture
