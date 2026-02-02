import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_53"
SUBMISSION_DIR = "./submission"

# Create necessary output directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Image Directory (Metadata contains relative paths like 'images/1.jpg')
# This base path is used to resolve those relative paths.
IMAGES_BASE_DIR = INPUT_DIR

# =============================================================================
# DATASET COLUMNS AND STRUCTURE
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"
IMAGE_PATH_COL = "image_path"

# Feature Definitions
# The dataset provides three sets of 64-attribute vectors
MARGIN_COLS = [f"margin_{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape_{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture_{i}" for i in range(1, 65)]

# Aggregated Feature Lists
ALL_PROVIDED_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# =============================================================================
# GLOBAL HYPERPARAMETERS
# =============================================================================
RANDOM_SEED = 42
VAL_SIZE = 0.2  # Consistent with the metadata split generation
PROB_CLIP = 1e-15  # Clipping value for log-loss metric stability

# =============================================================================
# MODEL EXPERT CONFIGURATION
# =============================================================================
# LDA Shrinkage Candidates
# A range of regularization strengths for the Linear Discriminant Analysis experts.
# 'auto' uses the Ledoit-Wolf lemma for automatic shrinkage estimation.
LDA_SHRINKAGE_CANDIDATES = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.6, 0.8, "auto"]

# Interaction Topology Configuration
# Number of discriminative components to retain before polynomial expansion
INTERACTION_N_COMPONENTS = 15
# Degree of polynomial features to generate (e.g., 2 for quadratic interactions)
INTERACTION_POLY_DEGREE = 2

# Image Processing Configuration
# Threshold for determining image polarity (foreground vs background) based on corner pixels
POLARITY_THRESHOLD = 0.5

# =============================================================================
# RUNTIME CONTROL
# =============================================================================
# Set MAX_TRAIN_SAMPLES to an integer (e.g., 100) to debug the pipeline with a subset of data.
# Set to None to use the full dataset.
MAX_TRAIN_SAMPLES = None
