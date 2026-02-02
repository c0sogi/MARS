import os
import torch


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_26"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # =========================================================================
    # Data Processing & Feature Engineering
    # =========================================================================
    # Temporal Window: t-5 to t+5 (Current + 5 past + 5 future = 11 frames)
    WINDOW_SIZE = 5
    TOTAL_FRAMES = (WINDOW_SIZE * 2) + 1

    # Physical Stability Constraints (Clamping)
    # Strictly clamp continuous inputs to prevent gradient explosions
    CLAMP_MIN = -100.0
    CLAMP_MAX = 100.0

    # Kinematic Features (Continuous)
    # These will be collected for Player 1 and Player 2 over the temporal window
    KINEMATIC_FEATURES = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",  # Signed acceleration
    ]

    # Derived Interaction Features (Computed per step)
    INTERACTION_FEATURES = ["distance", "relative_speed", "relative_angle"]

    # Visual Features (Helmet Boxes)
    # Used in the visual correction stream
    VISUAL_FEATURES = [
        "left",
        "top",
        "width",
        "height",
        "view_area",  # Area of the bounding box, used for max-pooling selection
    ]

    # Categorical Features for Embeddings
    CATEGORICAL_COLS = ["position", "team"]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Entity Embeddings
    # Dimensions: (Number of unique values, Embedding Size)
    # Positions: Approx 25-30 unique roles.
    # Teams: Home, Away, Ground (mapped to special token).
    EMBEDDING_DIMS = {"position": (32, 8), "team": (4, 2)}

    # Backbone Dimensions
    KINEMATIC_HIDDEN_DIM = 256
    VISUAL_HIDDEN_DIM = 64
    DROPOUT_RATE = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 1024  # Large batch size for stable gradients with Focal Loss
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # Focal Loss Parameters (Critical for Class Imbalance)
    # Alpha=0.25 balances easy negatives
    # Gamma=2.0 focuses on hard examples
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0

    # =========================================================================
    # Inference
    # =========================================================================
    # Threshold will be optimized on validation set, but define a default
    DEFAULT_THRESHOLD = 0.5
