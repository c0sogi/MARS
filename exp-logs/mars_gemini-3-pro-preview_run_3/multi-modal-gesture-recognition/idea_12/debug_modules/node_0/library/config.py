import os
import torch


class Config:
    """
    Configuration for the Structurally-Augmented Attentive Kinematic Network (SA-AKN).
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for Idea 12 (SA-AKN)
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Reproducibility
    SEED = 42

    # Windowing Strategy
    # "Sliding Windows of 64 frames"
    WINDOW_SIZE = 64
    # Stride for training data generation (smaller stride = more augmentation)
    WINDOW_STRIDE_TRAIN = 16
    # Stride for inference: "Sliding Window Inference with 50% overlap"
    WINDOW_STRIDE_TEST = 32

    # Input Features (Kinematically Consistent)
    NUM_JOINTS = 20
    # Features:
    # 1. Root-Relative Positions (3)
    # 2. Bone Vectors (3)
    # 3. Velocity (3)
    # 4. Acceleration (3)
    CHANNELS_PER_JOINT = 12
    # Input Dimension: 20 joints * 12 channels = 240
    INPUT_DIM = 240

    # Labels
    # Vocabulary of 20 gestures + 1 background class
    # IDs in dataset are 1-20. We map 0 to Background.
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # Model Architecture
    # ==========================================
    # Stage 1: Sequence Encoder (Bi-GRU)
    GRU_HIDDEN_DIM = 128
    GRU_LAYERS = 2

    # Stage 2 & 3: Attentive Gated Refinement (TCN + SE)
    TCN_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    TCN_LAYERS = 4
    TCN_DROPOUT = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # General
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Function Weights
    # "A weight of 0.2 is assigned to the 'background' class"
    BACKGROUND_WEIGHT = 0.2
    # Weight for the Log-Space Smoothing Loss (MSE on log-probs)
    SMOOTHING_LOSS_WEIGHT = 0.15

    # Optimization
    # "Use the Adam optimizer. We avoid AdamW"
    USE_ADAMW = False
    WEIGHT_DECAY = 0.0

    # ==========================================
    # Debug / Runtime Control
    # ==========================================
    # Set to True to use a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLES = 50

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_class_weights(cls):
        """
        Returns the class weights tensor for Weighted Cross-Entropy Loss.
        """
        weights = torch.ones(cls.NUM_CLASSES)
        weights[cls.BACKGROUND_CLASS_ID] = cls.BACKGROUND_WEIGHT
        return weights
