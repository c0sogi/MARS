import os
import torch


class Config:
    """
    Configuration for the Linear-Residual Pyramidal Invariant Network (LRP-Net) pipeline.
    Centralizes file paths, hyperparameters, and feature definitions.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_40"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated in previous steps)
    META_TRAIN = os.path.join(METADATA_DIR, "train.csv")
    META_VAL = os.path.join(METADATA_DIR, "validation.csv")
    META_TEST = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    TRAIN_TRACKING = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "lrp_net_model.pth")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Engineering Configuration
    # ==========================================
    # Temporal Window: t-5 to t+5 (Total 11 timesteps)
    WINDOW_SIZE = 5
    TOTAL_TIMESTEPS = WINDOW_SIZE * 2 + 1

    # Physical Constraints (Clamping)
    # Used to prevent outliers in derivative features from destabilizing gradients
    CLAMP_MIN = -50.0
    CLAMP_MAX = 50.0

    # Feature Definitions
    # Kinematic features to be lagged and windowed
    KINEMATIC_COLS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Visual features from helmet boxes (Endzone/Sideline/All29)
    # These will be processed via Max-Pooling Selection Strategy
    VISUAL_COLS = ["left", "width", "top", "height"]

    # Target Variable
    TARGET_COL = "contact"

    # Dataset Management
    # Set DEBUG=True to use a small subset of data for rapid prototyping
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000

    # ==========================================
    # Model Architecture (LRP-Net)
    # ==========================================
    # Pyramidal Backbone Dimensions (Interleaved ResBlocks)
    # Structure: Input -> 512 -> 256 -> 128 -> Logits
    PYRAMID_DIMS = [512, 256, 128]

    # Visual Stream Architecture
    # Shallow MLP for visual correction
    VISUAL_HIDDEN_DIM = 64

    # Fusion Parameters
    # Logit_final = L_linear + L_deep + lambda * L_vis
    VISUAL_LAMBDA = 0.5

    # Regularization
    # Gaussian noise injection sigma for kinematic features during training
    INPUT_NOISE_SIGMA = 0.05
    DROPOUT_RATE = 0.1

    # ==========================================
    # Training Configuration
    # ==========================================
    # Large batch size for stable BatchNorm statistics
    BATCH_SIZE = 8192

    # Optimization
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 3

    # Focal Loss Parameters
    # Optimized for class imbalance (approx 1:72)
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # ==========================================
    # Hardware Configuration
    # ==========================================
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_directories():
        """
        Creates the necessary working and submission directories if they do not exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
