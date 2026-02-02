import os
import torch

# ==========================================
# 1. Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific cache directory for this strategy (Idea 7)
CACHE_DIR = os.path.join(WORKING_DIR, "optimized")

# Raw Data Paths
TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. Global Configuration
# ==========================================
SEED = 42
DEBUG = False  # Set to True to run on a small subset for testing
NUM_WORKERS = 4  # Number of DataLoader workers

# ==========================================
# 3. Data & Feature Engineering
# ==========================================
# Sequence length is fixed by the breath duration in the dataset (~80 steps)
# We pad/truncate to this length if necessary, though data is mostly uniform.
MAX_SEQ_LEN = 80

# Feature definitions
# These lists guide the FeatureEngineering pipeline

# Continuous features to be scaled (RobustScaler recommended)
# Includes raw physics parameters and engineered derivatives/integrals
CONTINUOUS_FEATURES = [
    "time_step",
    "u_in",
    "R",
    "C",
    # Integral / Volume approximation
    "u_in_cumsum",
    # Physics Interactions
    "R_u_in",  # Resistive pressure component (Flow * Resistance)
    "u_in_cumsum_div_C",  # Elastic pressure component (Volume / Compliance)
    # Temporal Derivatives (Velocity/Acceleration of control)
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_lag4",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_diff3",
    "u_in_diff4",
]

# Categorical / Binary features
# u_out is the expiratory valve status (0 or 1)
CATEGORICAL_FEATURES = ["u_out"]

# Target column
TARGET_COL = "pressure"

# ID columns
ID_COL = "id"
BREATH_ID_COL = "breath_id"

# ==========================================
# 4. Model Hyperparameters (DPI-BiLSTM)
# ==========================================
# Architecture specifics for Deep Projected-Injection BiLSTM
INPUT_DIM = len(CONTINUOUS_FEATURES) + len(
    CATEGORICAL_FEATURES
)  # Calculated dynamically
PROJECTION_DIM = 512  # Dimension of the Latent Input Projection
HIDDEN_DIM = 512  # Hidden dimension of the LSTM units
NUM_LSTM_LAYERS = 4  # Deep stack (4-6 layers recommended)
DROPOUT = 0.1  # Dropout rate for regularization
USE_LAYER_NORM = True  # Layer Normalization for stability

# ==========================================
# 5. Training Hyperparameters
# ==========================================
# Long-tail convergence strategy requires extended epochs
EPOCHS = 150 if not DEBUG else 2
BATCH_SIZE = 512 if not DEBUG else 64

# Optimizer (AdamW)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05  # Increased regularization for deep recurrent stability
EPSILON = 1e-6

# Scheduler (Cosine Annealing)
T_MAX = EPOCHS  # For CosineAnnealingLR
ETA_MIN = 1e-6  # Minimum learning rate

# Loss Function Weights
# Weighted L1 Loss: Focus on Inspiratory phase (u_out=0)
INSPIRATORY_WEIGHT = 1.0
EXPIRATORY_WEIGHT = 0.1

# Gradient Clipping
MAX_GRAD_NORM = 1000.0

# ==========================================
# 6. Hardware
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
