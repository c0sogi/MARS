import os

# =============================================================================
# Path Configuration
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_24"
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# Global Hyperparameters
# =============================================================================
SEED = 42
WINDOW_SIZE = 5  # Temporal window: captures frames from t-5 to t+5 (11 frames total)

# =============================================================================
# Training Hyperparameters
# =============================================================================
# Focal Loss parameters for class imbalance handling
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Optimization parameters
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 3  # Early stopping patience

# =============================================================================
# Feature Configuration
# =============================================================================

# Categorical Features: Used for Entity Embeddings in the Kinematic Stream
# These capture role-based physics priors (e.g., QB vs LB behavior)
CATEGORICAL_COLS = ["position", "team"]

# Kinematic Features: Continuous variables from player tracking data
# These serve as the base for the Context-Aware Kinematic Backbone
# They will be windowed (t-WINDOW_SIZE to t+WINDOW_SIZE) and flattened
KINEMATIC_COLS = [
    "x_position",
    "y_position",
    "speed",
    "distance",
    "direction",
    "orientation",
    "acceleration",
    "sa",
]

# Visual Features: Geometric metrics from helmet bounding boxes
# Used in the Visual Correction Stream (Shallow MLP)
VISUAL_COLS = ["left", "width", "top", "height"]

# Reliability Gating Features: Metadata to gate the visual stream
# Used to dynamically suppress visual features when quality is low
VISUAL_META_COLS = [
    "view_available",  # Binary flag indicating if a valid view was found
    "box_area",  # Proxy for confidence/proximity
]

# Target Variable
TARGET_COL = "contact"

# =============================================================================
# Data Processing Constraints (Numerical Stability)
# =============================================================================
# Clamping bounds to prevent exploding gradients from derivative features
CLAMP_MIN = -50.0
CLAMP_MAX = 50.0

# Ground Imputation Constants
GROUND_SPEED = 0.0
GROUND_ACCEL = 0.0
