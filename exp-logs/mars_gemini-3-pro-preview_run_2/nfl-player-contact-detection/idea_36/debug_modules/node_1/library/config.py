import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_36"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data Paths (for reference by data loaders)
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
NUM_WORKERS = 4  # Adjust based on vCPU availability (12 vCPUs available)

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
# Temporal Window: t-5 to t+5 (0.5 seconds context)
WINDOW_SIZE = 5

# Kinematic Features (Tracking Data)
# These are the raw columns from tracking data used to generate the flattened wide vector.
# The model will use lags of these features.
KINEMATIC_FEATURES = [
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "direction",
    "orientation",
    "sa",  # Signed acceleration
]

# Visual Features (Helmet Data)
# These are the bounding box metrics used in the visual stream.
VISUAL_FEATURES = ["left", "width", "top", "height"]

# Meta Columns for identification
META_COLUMNS = ["game_play", "step", "nfl_player_id"]

# Physical Constraints for Clamping (Stability)
# Used to prevent gradient explosions from outliers
CLAMP_CONFIG = {
    "speed": (-50.0, 50.0),
    "acceleration": (-50.0, 50.0),
    "sa": (-50.0, 50.0),
    "distance": (0.0, 150.0),  # Field size constraint
}

# =============================================================================
# MODEL HYPERPARAMETERS (NR-PIRV-Net)
# =============================================================================
# Pyramidal Backbone Dimensions for the Kinematic Stream
# Structure: Input -> 512 -> 256 -> 128 -> Logit
PYRAMID_DIMS = [512, 256, 128]

# Dimension for the Shallow Visual Stream MLP
VISUAL_HIDDEN_DIM = 64

# Input Noise Injection (Structural Regularization)
# Standard deviation of Gaussian noise added to kinematic inputs during training
NOISE_SIGMA = 0.05

# Dropout Rate for Residual Blocks
DROPOUT_RATE = 0.2

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Large batch size for stable BatchNorm statistics in Pyramidal backbone
BATCH_SIZE = 8192

# Optimizer Settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Training Duration
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3

# Focal Loss Parameters
# alpha=0.25 balances positive/negative classes
# gamma=2.0 focuses on hard examples
FOCAL_LOSS_PARAMS = {"alpha": 0.25, "gamma": 2.0}

# =============================================================================
# INFERENCE & EVALUATION
# =============================================================================
# Threshold optimization range for MCC maximization
THRESHOLD_SEARCH_START = 0.1
THRESHOLD_SEARCH_END = 0.9
THRESHOLD_SEARCH_STEPS = 100
