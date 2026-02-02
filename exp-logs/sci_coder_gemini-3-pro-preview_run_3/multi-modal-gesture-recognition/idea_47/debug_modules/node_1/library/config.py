import os
import torch


class Config:
    """
    Configuration class for the Decoupled-Norm Gated Central-Kinematic Network (DGC-KN).
    Centralizes all hyperparameters, file paths, and execution settings.
    """

    # ==========================================
    # Reproducibility & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and checkpoints
    # Idea 47 represents the current iteration: DGC-KN
    WORKING_DIR = "./working/idea_47"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache file paths (using .npz for efficient storage)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.npz")

    # Model checkpoint path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Windowing strategy for temporal segmentation
    WINDOW_SIZE = 64
    STRIDE = 32

    # Class definitions
    # 20 Gestures + 1 Background class (Index 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

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

    # Feature Extraction
    SKELETON_JOINTS = 20
    SKELETON_CHANNELS = 3  # X, Y, Z

    # Input Feature Dimensions
    # Skeleton: (Pos + Vel + Acc) * Joints * Channels = 3 * 20 * 3 = 180
    # Audio: 13 MFCC coefficients
    INPUT_DIM_SKELETON = 180
    INPUT_DIM_AUDIO = 13
    TOTAL_INPUT_DIM = INPUT_DIM_SKELETON + INPUT_DIM_AUDIO

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Stage 1: Bi-GRU Encoder
    HIDDEN_DIM = 192  # 96 units per direction * 2
    GRU_LAYERS = 2
    DROPOUT = 0.4

    # Stage 2 & 3: TCN Refinement
    # Monotonically increasing dilation to match RF of ~63 frames
    TCN_KERNEL_SIZE = 3
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    TCN_CHANNELS = 192  # Matches hidden dim

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Loss Function Configuration
    # Weighted Cross Entropy: Downweight background class to focus on gestures
    LOSS_BG_WEIGHT = 0.2
    LOSS_GESTURE_WEIGHT = 1.0

    # Log-Space Smoothing Loss (Calibration)
    SMOOTHING_LAMBDA = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # Deep Supervision Weights
    # L_total = W1*L(P1) + W2*L(P2) + W3*L(P3)
    LOSS_STAGE_WEIGHTS = [1.0, 1.0, 1.0]

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    # Minimum duration (in frames) to keep a predicted gesture segment
    MIN_GESTURE_DURATION = 5

    # Debugging / Development
    # Set to a small integer (e.g., 100) to limit dataset size for rapid testing
    # Set to None for full training
    DEBUG_SUBSET_SIZE = None

    @classmethod
    def set_debug_mode(cls, subset_size=100, epochs=2):
        """
        Helper to switch configuration to debug mode.
        """
        cls.DEBUG_SUBSET_SIZE = subset_size
        cls.NUM_EPOCHS = epochs
        print(
            f"Config configured for DEBUG mode: Subset={subset_size}, Epochs={epochs}"
        )

    @classmethod
    def get_class_weights(cls):
        """
        Returns the weight tensor for CrossEntropyLoss.
        """
        weights = torch.ones(cls.NUM_CLASSES)
        weights[cls.BACKGROUND_CLASS_ID] = cls.LOSS_BG_WEIGHT
        return weights.to(cls.DEVICE)
