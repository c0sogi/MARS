import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_51"

    # Ensure working directory exists for caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache paths
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_cache.npz")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_cache.npz")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_cache.npz")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # Data Pipeline Parameters
    # ==========================================
    # Windowing
    WINDOW_SIZE = 64
    STRIDE = 32

    # Skeleton Structure
    NUM_JOINTS = 20
    CHANNELS_PER_JOINT = 3  # X, Y, Z

    # Explicit Bone Connections (Kinect Format Indices 0-19)
    # Used to calculate structural bone vectors
    BONE_PAIRS = [
        (0, 1),  # HipCenter -> Spine
        (1, 2),  # Spine -> ShoulderCenter
        (2, 3),  # ShoulderCenter -> Head
        (2, 4),  # ShoulderCenter -> ShoulderLeft
        (4, 5),  # ShoulderLeft -> ElbowLeft
        (5, 6),  # ElbowLeft -> WristLeft
        (6, 7),  # WristLeft -> HandLeft
        (2, 8),  # ShoulderCenter -> ShoulderRight
        (8, 9),  # ShoulderRight -> ElbowRight
        (9, 10),  # ElbowRight -> WristRight
        (10, 11),  # WristRight -> HandRight
        (0, 12),  # HipCenter -> HipLeft
        (12, 13),  # HipLeft -> KneeLeft
        (13, 14),  # KneeLeft -> AnkleLeft
        (14, 15),  # AnkleLeft -> FootLeft
        (0, 16),  # HipCenter -> HipRight
        (16, 17),  # HipRight -> KneeRight
        (17, 18),  # KneeRight -> AnkleRight
        (18, 19),  # AnkleRight -> FootRight
    ]
    NUM_BONES = len(BONE_PAIRS)  # 19

    # Audio
    AUDIO_N_MFCC = 13

    # Robustness & Augmentation
    NOISE_SIGMA = 0.01  # Gaussian noise injection before derivation
    MIN_GESTURE_FRAMES = 5  # Filter short predictions

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    # Input Dimension Calculation:
    # 1. Position (Centered): 20 joints * 3 = 60
    # 2. Bone Vectors: 19 bones * 3 = 57
    # 3. Velocity: 20 joints * 3 = 60
    # 4. Acceleration: 20 joints * 3 = 60
    # 5. Audio: 13 MFCCs
    # Total = 60 + 57 + 60 + 60 + 13 = 250
    INPUT_DIM = 250

    # Encoder (Stage 1)
    HIDDEN_DIM = 192
    GRU_LAYERS = 2
    DROPOUT = 0.4

    # Refinement (Stage 2 & 3)
    TCN_KERNEL_SIZE = 3
    TCN_DILATIONS = [1, 2, 4, 8, 16]  # Receptive field approx 63
    USE_SE_BLOCK = True  # Squeeze-and-Excitation for global context

    # Classes: 20 gestures + 1 background (0)
    NUM_CLASSES = 21

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Small batch size for sequence data
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Loss Configuration
    # Weighted Cross Entropy
    BACKGROUND_CLASS_WEIGHT = 0.2
    ACTIVE_CLASS_WEIGHT = 1.0

    # Log-Space Smoothing Loss (Truncated MSE on log-probs)
    SMOOTHING_LOSS_WEIGHT = 0.15
    SMOOTHING_LOSS_THRESHOLD = 1.0

    @classmethod
    def get_class_weights(cls):
        """Returns tensor of class weights."""
        weights = torch.ones(cls.NUM_CLASSES)
        weights[0] = cls.BACKGROUND_CLASS_WEIGHT
        return weights.to(cls.DEVICE)
