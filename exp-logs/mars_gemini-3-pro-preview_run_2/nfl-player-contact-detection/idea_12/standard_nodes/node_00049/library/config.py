import os
import torch


class Config:
    """
    Configuration for Entity-Centric Physics-Informed Residual Network (EC-PIRN).
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Create working directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Pipeline Hyperparameters
    # =========================================================================
    # Window size: t-5 to t+5 (inclusive) -> 11 timesteps total
    WINDOW_SIZE = 5

    # Raw tracking columns to extract from source files
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Features used per timestep in the flattened input vector
    # 1. Player 1 features (7)
    P1_FEATURES = [f"{col}_1" for col in TRACKING_COLS]

    # 2. Player 2 features (7) - Ground values imputed as 0/Player1-pos where appropriate
    P2_FEATURES = [f"{col}_2" for col in TRACKING_COLS]

    # 3. Interaction/Physics features (4)
    # distance: Euclidean distance
    # log_distance: np.log1p(distance) for resolution near 0
    # relative_speed: Magnitude of velocity difference vector
    # closing_speed: Rate of change of distance
    INTERACTION_FEATURES = [
        "distance",
        "log_distance",
        "relative_speed",
        "closing_speed",
    ]

    # 4. Meta features (1)
    META_FEATURES = ["is_ground"]

    # Total features per timestep
    FEATURES_PER_STEP = P1_FEATURES + P2_FEATURES + INTERACTION_FEATURES + META_FEATURES

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Input dimension = (Features per step) * (Total steps in window)
    INPUT_DIM = len(FEATURES_PER_STEP) * (2 * WINDOW_SIZE + 1)

    HIDDEN_DIM = 512
    NUM_RESIDUAL_BLOCKS = 4
    DROPOUT_RATE = 0.2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3

    # Focal Loss Parameters
    # Alpha: Balance factor for class imbalance (0.25 typically downweights the majority class 0)
    # Gamma: Focusing parameter to penalize hard-to-classify examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def get_feature_names(cls):
        """
        Returns the list of feature names for the flattened vector.
        Useful for debugging or feature importance analysis.
        """
        feature_names = []
        # Steps range from -WINDOW_SIZE to +WINDOW_SIZE
        for t in range(-cls.WINDOW_SIZE, cls.WINDOW_SIZE + 1):
            suffix = f"_t{t}" if t < 0 else f"_t+{t}"
            # Handle t=0 case or just sign
            if t == 0:
                suffix = "_t0"

            for feat in cls.FEATURES_PER_STEP:
                feature_names.append(f"{feat}{suffix}")
        return feature_names
