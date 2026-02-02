import os

# =============================================================================
# Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_16")
SUBMISSION_DIR = "./submission"

# Ensure necessary writeable directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# File Paths
# =============================================================================
# Metadata paths (Stratified splits)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw input paths
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Hyperparameters & Constants
# =============================================================================
RANDOM_SEED = 42

# Semi-supervised learning threshold
# Only test samples with max probability > this value will be used for covariance refinement
CONFIDENCE_THRESHOLD = 0.99

# Log Loss Clipping limits (as per metric definition)
# max(min(p, 1-10^-15), 10^-15)
PROB_CLIP_MIN = 1e-15
PROB_CLIP_MAX = 1.0 - 1e-15

# =============================================================================
# Data Definitions
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"

# Feature Lists
# We explicitly define these to ensure deterministic ordering of columns
# in the numpy arrays used for training and inference.
MARGIN_FEATURES = [f"margin_{i}" for i in range(1, 65)]
SHAPE_FEATURES = [f"shape_{i}" for i in range(1, 65)]
TEXTURE_FEATURES = [f"texture_{i}" for i in range(1, 65)]

# Combined Feature List (Total 192 features)
# Order: Margin -> Shape -> Texture
FEATURE_COLS = MARGIN_FEATURES + SHAPE_FEATURES + TEXTURE_FEATURES
