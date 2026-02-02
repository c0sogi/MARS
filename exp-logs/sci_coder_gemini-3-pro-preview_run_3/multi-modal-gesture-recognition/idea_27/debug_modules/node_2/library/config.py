import os
import torch


class Config:
    """
    Configuration for the Gated High-Capacity Kinematic Refinement Network (GHC-KRN).
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (Idea 27)
    WORKING_DIR = "./working/idea_27"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model checkpoints
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure critical directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Sliding Window Strategy
    WINDOW_SIZE = 64
    STRIDE_TRAIN = 32
    STRIDE_TEST = 32  # 50% overlap for inference

    # Input Feature Dimensions
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3  # X, Y, Z

    # Kinematic Augmentation Flags
    USE_VELOCITY = True
    USE_ACCELERATION = True

    # Audio Features
    AUDIO_N_MFCC = 13

    # Calculated Input Dimension
    # Raw (60) + Velocity (60) + Acceleration (60) = 180
    SKELETON_DIM = (
        NUM_JOINTS * COORDS_PER_JOINT * (1 + int(USE_VELOCITY) + int(USE_ACCELERATION))
    )
    # Total Input = 180 + 13 = 193
    INPUT_DIM = SKELETON_DIM + AUDIO_N_MFCC

    # Class Definitions
    # 20 gestures + 1 background class
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # Label Map (Name -> ID)
    # Note: Dataset labels are 1-20. We map them to 1-20 in the model output indices.
    # Index 0 is reserved for Background.
    LABEL_MAP = {
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

    # Inverse map for submission generation
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

    # ==========================================
    # Model Architecture (GHC-KRN)
    # ==========================================
    # Stage 1: Gated High-Capacity Kinematic Encoder (Bi-GRU)
    # 128 units per direction * 2 directions = 256 total hidden size
    ENCODER_HIDDEN_DIM = 256
    GRU_NUM_LAYERS = 1
    DROPOUT = 0.3

    # Stage 2 & 3: Monotonic Non-Causal MS-TCN
    MSTCN_STAGES = 2  # Number of refinement stages (Stage 2 and Stage 3)
    MSTCN_LAYERS = 5  # Layers per stage, corresponding to dilation [1, 2, 4, 8, 16]
    MSTCN_FILTERS = 64  # Feature channels in TCN
    MSTCN_KERNEL_SIZE = 3  # Kernel size

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Adam default
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Function Configuration
    # Background class gets lower weight (0.2) to focus on gestures
    LOSS_WEIGHT_BACKGROUND = 0.2

    # Log-Space Smoothing Loss (Truncated MSE)
    WEIGHT_SMOOTHING = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    # Minimum duration to consider a segment valid (in frames)
    MIN_GESTURE_DURATION = 5

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_class_weights():
        """
        Returns the class weight tensor for the Loss function.
        Weights: 0.2 for background (index 0), 1.0 for all others.
        """
        weights = torch.ones(Config.NUM_CLASSES)
        weights[Config.BACKGROUND_CLASS_ID] = Config.LOSS_WEIGHT_BACKGROUND
        return weights.to(Config.DEVICE)
