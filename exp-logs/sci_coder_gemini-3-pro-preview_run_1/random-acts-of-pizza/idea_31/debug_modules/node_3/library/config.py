import os

# =============================================================================
# GLOBAL PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_31"
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# DATA & FEATURE ENGINEERING CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
VAL_SIZE = 0.2

# Text Processing / Embeddings
TFIDF_VOCAB = 5000  # High-fidelity vocabulary size for RF
SBERT_MODEL = "all-MiniLM-L6-v2"  # Model for generating semantic embeddings

# Feature Flags & Dimensions
TOP_K_SUBREDDITS = 50  # Number of top frequent subreddits to encode as binary flags
USE_DUAL_VIEW_ALIGNMENT = True  # Enable Topic vs Narrative consistency features
USE_ARCSINH_TRANSFORM = True  # Apply arcsinh transform to skewed metadata

# Debugging / Development
DEBUG = False  # Set to True to enable quick debugging runs
MAX_SAMPLES = None  # Limit number of samples (e.g., 100) if DEBUG is True

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Stream A: Dual-View Consistency Random Forest
# -----------------------------------------------------------------------------
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 1  # Minimal regularization to preserve sparse Top-K signals
RF_CLASS_WEIGHT = "balanced"  # Handle class imbalance
RF_MAX_DEPTH = None  # Allow full tree growth
RF_N_JOBS = -1  # Use all available cores

# Stream B: Simplified Dual-Query Alignment-Injected MLP
# -----------------------------------------------------------------------------
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.5  # Robust dropout for regularization
MLP_LR = 1e-4
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE = 32
MLP_EPOCHS = 50
MLP_PATIENCE = 15  # High patience to allow dual-attention stabilization
MLP_USE_BATCHNORM = False  # Disabled to prevent overfitting (Lesson 00062)

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Weights for the final weighted average [Random Forest, MLP]
# Using 0.5/0.5 to ensure robustness and prevent collapse
ENSEMBLE_WEIGHTS = [0.5, 0.5]
