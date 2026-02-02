import os
import torch


class Config:
    """
    Global configuration for the Residual-Visual Corrected Kinematic Network (RVC-Net).
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Data Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Artifact Paths
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    MODEL_PATH = os.path.join(WORKING_DIR, "rvc_net_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Preprocessing & Feature Engineering
    # =========================================================================
    # Window size: t-K to t+K. Total window length = 2*K + 1
    # Idea specifies t-5 to t+5
    WINDOW_K = 5

    # --- Kinematic Stream Features ---
    # Raw columns to extract from tracking CSV
    TRACKING_RAW_COLS = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Features used per player (P1 and P2) in the model input
    KINEMATIC_PLAYER_COLS = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Features calculated between the pair (Relative Physics)
    KINEMATIC_PAIR_COLS = [
        "distance",
        "log_distance",  # np.log1p(distance)
        "relative_speed",
        "relative_acceleration",  # New: Explicit relative physics (Cite Lesson 00033)
        "orientation_diff",  # New: Explicit relative physics (Cite Lesson 00033)
        "clamped_closing_speed",  # specialized closing speed feature
    ]

    # --- Visual Stream Features ---
    # Raw columns to extract from helmet CSV
    HELMET_RAW_COLS = ["left", "top", "width", "height"]

    # Features used per player (P1 and P2)
    # Visual stream learns from box geometry (e.g. box_top indicates proximity to camera/ground)
    VISUAL_PLAYER_COLS = ["left", "top", "width", "height"]

    # Features calculated between the pair
    VISUAL_PAIR_COLS = [
        "helmet_iou",
        "helmet_centroid_dist",
        "log_helmet_centroid_dist",  # New: Non-linear transform (Cite Lesson 00005)
    ]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Kinematic Stream (Deep Residual MLP)
    # Input dim is calculated dynamically below
    KIN_HIDDEN_DIMS = [512, 256, 128]
    KIN_DROPOUT = 0.2

    # Visual Stream (Shallow MLP)
    VIS_HIDDEN_DIMS = [64, 32]
    VIS_DROPOUT = 0.1

    # Residual Fusion
    # Logit_final = Logit_kin + lambda * Logit_vis
    # Initial value for lambda if learnable, or fixed value
    RESIDUAL_LAMBDA_INIT = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 4096  # Large batch size for efficient tabular training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Focal Loss Parameters (for 1:72 class imbalance)
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Threshold Optimization
    # Range of thresholds to search for MCC maximization
    THRESHOLD_SEARCH_START = 0.1
    THRESHOLD_SEARCH_END = 0.9
    THRESHOLD_SEARCH_STEP = 0.01

    # =========================================================================
    # Helper Methods
    # =========================================================================
    @classmethod
    def get_kinematic_input_dim(cls):
        """
        Calculates the flattened input dimension for the Kinematic Stream.
        (Features_P1 + Features_P2 + Features_Pair) * Window_Size
        """
        n_p1 = len(cls.KINEMATIC_PLAYER_COLS)
        n_p2 = len(cls.KINEMATIC_PLAYER_COLS)
        n_pair = len(cls.KINEMATIC_PAIR_COLS)
        window_len = 2 * cls.WINDOW_K + 1
        return (n_p1 + n_p2 + n_pair) * window_len

    @classmethod
    def get_visual_input_dim(cls):
        """
        Calculates the flattened input dimension for the Visual Stream.
        (Features_P1 + Features_P2 + Features_Pair) * Window_Size
        """
        n_p1 = len(cls.VISUAL_PLAYER_COLS)
        n_p2 = len(cls.VISUAL_PLAYER_COLS)
        n_pair = len(cls.VISUAL_PAIR_COLS)
        window_len = 2 * cls.WINDOW_K + 1
        return (n_p1 + n_p2 + n_pair) * window_len
