import os
import torch


class Config:
    """
    Configuration for the Time-Distributed Stabilized Residual Network (TD-SRN) solution.
    Centralizes paths, hyperparameters, and numerical stability constraints.
    """

    # =========================================================================
    # File System Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = "./working/idea_28/"
    SUBMISSION_DIR = "./submission"

    # Metadata Splits (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Sources
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Outputs
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(CACHE_DIR, "td_srn_best_model.pth")

    # Cache Files (Parquet)
    CACHE_TRAIN_FEATURES = os.path.join(CACHE_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(CACHE_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(CACHE_DIR, "test_features.parquet")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Processing & Feature Engineering
    # =========================================================================
    # Temporal Window: t-5 to t+5 (11 frames total at 10Hz)
    WINDOW_SIZE = 11
    WINDOW_HALF = 5

    # Explicit Numerical Stability (Cite solution_lesson_node_00081)
    # Clamping range for derived kinematic features to prevent outliers
    CLAMP_MIN = -50.0
    CLAMP_MAX = 50.0

    # Feature Flags
    USE_LOG_DISTANCE = True

    # =========================================================================
    # Model Architecture (TD-SRN)
    # =========================================================================
    # Dimensions
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.1

    # Residual Fusion
    # Weight for adding visual logit stream to kinematic stream
    VISUAL_LAMBDA = 1.0

    # =========================================================================
    # Training Configuration
    # =========================================================================
    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Optimization
    BATCH_SIZE = 1024  # Large batch size allowed by efficient architecture
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # For AdamW
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # Loss Function: Focal Loss (Cite solution_lesson_node_00077)
    # Alpha balances positive/negative class importance
    # Gamma focuses on hard examples
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0

    @staticmethod
    def setup_directories():
        """
        Ensures that the necessary working and submission directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
