import os
import torch


class Config:
    """
    Configuration for the Residual Log-Kinematic Refinement Network (RLK-RN).
    Centralizes hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific solution (Idea 22)
    WORK_DIR = "./working/idea_22"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    MODEL_DIR = os.path.join(WORK_DIR, "model")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Artifact Paths
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # 3. Data Processing & Augmentation
    # ==========================================
    # Skeleton Data
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3  # X, Y, Z
    USE_VELOCITY = True
    USE_ACCELERATION = True

    # Audio Data
    AUDIO_N_MFCC = 13
    AUDIO_SAMPLE_RATE = 16000

    # Input Dimensionality Calculation
    # Structure: [Position (60) | Velocity (60) | Acceleration (60) | Audio (13)]
    # Total: 193 features per frame
    INPUT_DIM = (NUM_JOINTS * COORDS_PER_JOINT) * 3 + AUDIO_N_MFCC

    # Sliding Window Strategy
    WINDOW_SIZE = 64
    STRIDE = 32

    # Labels
    # Classes 1-20 are gestures, 0 is background
    NUM_CLASSES = 21
    BACKGROUND_LABEL = 0

    # ==========================================
    # 4. Model Architecture (RLK-RN)
    # ==========================================
    # Stage 1: Wide-Capacity Kinematic Encoder
    # Bi-GRU: 128 units per direction -> 256 total output dim
    ENCODER_HIDDEN_DIM = 128
    ENCODER_NUM_LAYERS = 2
    ENCODER_DROPOUT = 0.3

    # Stage 2 & 3: Residual Sawtooth Refinement
    # TCN (Temporal Convolutional Network) settings
    TCN_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # Dilation Schedule: Repeated Sawtooth to maximize receptive field while preserving local detail
    # Total Receptive Field must be covered effectively within WINDOW_SIZE
    DILATION_SCHEDULE = [1, 2, 4, 8, 1, 2, 4, 8]

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Training Loop
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Deep Supervision Weights
    # Sum of losses from Stage 1, Stage 2, and Stage 3
    LOSS_WEIGHTS = {"stage1": 1.0, "stage2": 1.0, "stage3": 1.0}

    # Class Balancing
    # Weight 0.2 for background to encourage gesture detection, 1.0 for all gesture classes
    CLASS_WEIGHTS = [0.2] + [1.0] * 20

    # Log-Space Smoothing Loss (Truncated MSE)
    SMOOTHING_LAMBDA = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # ==========================================
    # 6. Development / Debugging
    # ==========================================
    # Set to True to run on a small subset of data for rapid testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories if they do not exist.
        """
        for d in [cls.WORK_DIR, cls.CACHE_DIR, cls.MODEL_DIR, cls.SUBMISSION_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def get_class_weights_tensor(cls, device):
        """
        Returns the class weights as a torch tensor on the specified device.
        """
        return torch.tensor(cls.CLASS_WEIGHTS, dtype=torch.float32).to(device)
