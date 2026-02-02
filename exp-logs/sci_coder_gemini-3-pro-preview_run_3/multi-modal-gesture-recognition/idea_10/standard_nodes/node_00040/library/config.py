import os
import torch


class Config:
    """
    Configuration class for the View-Invariant Attentive Refinement Network (VI-ARN).
    Centralizes all hyperparameters, file paths, and constants.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Create necessary writable directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Cache Directory for processed features
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Model and Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Processing Hyperparameters
    # ==========================================
    # Skeleton Configuration
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3  # X, Y, Z

    # Feature Engineering Flags
    USE_CANONICAL_ALIGNMENT = True  # Align hips to X-axis
    USE_KINEMATIC_FEATURES = True  # Include Velocity and Acceleration

    # Audio Configuration
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13

    # Windowing Strategy
    WINDOW_SIZE = 64
    STRIDE = 32  # 50% overlap

    # Input Dimension Calculation
    # Skeleton: 20 joints * 3 coords * 3 (Pos + Vel + Acc) = 180
    SKELETON_FEAT_DIM = (
        NUM_JOINTS * COORDS_PER_JOINT * (3 if USE_KINEMATIC_FEATURES else 1)
    )
    AUDIO_FEAT_DIM = N_MFCC

    # Total Input Dimension for the Model
    INPUT_DIM = SKELETON_FEAT_DIM + AUDIO_FEAT_DIM

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    MODEL_NAME = "VI-ARN"

    # Stage 1: Bi-GRU
    GRU_HIDDEN_DIM = 128
    GRU_NUM_LAYERS = 2

    # Stage 2 & 3: TCN
    TCN_NUM_CHANNELS = [64, 64, 64]
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3

    # Classification
    # 20 Gestures + 1 Background
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50

    # Loss Function Weights
    # Weight for the background class (Class 0) to handle imbalance
    BACKGROUND_LOSS_WEIGHT = 0.2

    # Weight for Log-Space Smoothing Loss (lambda_smooth)
    LAMBDA_SMOOTH = 0.15

    # Optimization
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0

    # ==========================================
    # 5. Debugging & Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # 6. Labels & Mappings
    # ==========================================
    # Ordered list of 20 gestures
    GESTURE_LABELS = [
        "vattene",
        "vieniqui",
        "perfetto",
        "furbo",
        "cheduepalle",
        "chevuoi",
        "daccordo",
        "seipazzo",
        "combinato",
        "freganiente",
        "ok",
        "cosatifarei",
        "basta",
        "prendere",
        "noncenepiu",
        "fame",
        "tantotempo",
        "buonissimo",
        "messidaccordo",
        "sonostufo",
    ]

    # Map Name -> ID (1-20)
    NAME_TO_ID = {name: i + 1 for i, name in enumerate(GESTURE_LABELS)}

    # Map ID -> Name (0 is Background)
    ID_TO_NAME = {i + 1: name for i, name in enumerate(GESTURE_LABELS)}
    ID_TO_NAME[0] = "background"

    # ==========================================
    # 7. Hardware & Utilities
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def get_class_weights(cls):
        """
        Returns the class weights tensor for CrossEntropyLoss.
        Background class gets weight 0.2, others get 1.0.
        """
        weights = torch.ones(cls.NUM_CLASSES)
        weights[cls.BACKGROUND_CLASS_ID] = cls.BACKGROUND_LOSS_WEIGHT
        return weights.to(cls.DEVICE)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print(f"Configuration: {cls.MODEL_NAME}")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(
            f"Input Dim: {cls.INPUT_DIM} (Skeleton: {cls.SKELETON_FEAT_DIM}, Audio: {cls.AUDIO_FEAT_DIM})"
        )
        print(f"Window Size: {cls.WINDOW_SIZE}, Stride: {cls.STRIDE}")
        print(
            f"Classes: {cls.NUM_CLASSES} (Background Weight: {cls.BACKGROUND_LOSS_WEIGHT})"
        )
        print(
            f"Training: {cls.NUM_EPOCHS} Epochs, BS={cls.BATCH_SIZE}, LR={cls.LEARNING_RATE}"
        )
        print(f"Debug Mode: {cls.DEBUG}")
        print("=" * 30)
