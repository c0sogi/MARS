import os
import torch

# ---------------------------------------------------------
# Directory & File Paths
# ---------------------------------------------------------
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"

# Create working directories if they don't exist
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
MODEL_DIR = WORKING_DIR  # Where to save the model weights

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Files
SENSOR_GEOMETRY_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Generated previously)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")

# Output Files
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ---------------------------------------------------------
# Data Preprocessing Hyperparameters
# ---------------------------------------------------------
# Pulse Sampling
NUM_POINTS = 128  # Number of pulses to sample per event (Nodes in the graph)
MIN_PULSES = 8  # Minimum pulses required to process an event (otherwise pad)

# Graph Construction
K_NEIGHBORS = 6  # Number of nearest neighbors for dynamic edge construction (KNN)

# Debugging / Development
# Set to a small integer (e.g., 1000) to limit dataset size for quick testing.
# Set to None to use the full dataset.
DEBUG_SAMPLE_SIZE = None

# ---------------------------------------------------------
# Model Architecture Hyperparameters
# ---------------------------------------------------------
# Input Dimensions
NODE_FEAT_DIM = 5  # Features: [x, y, z, time, charge] (normalized)
EDGE_FEAT_DIM = 4  # Features: [dx, dy, dz, dt]

# Network Dimensions
HIDDEN_DIM = 128  # Hidden dimension for GNN layers and MLPs
GLOBAL_POOL_DIM = 256  # Dimension after global pooling
EIGEN_FEAT_DIM = (
    9  # 3 eigenvalues + 6 unique elements of covariance matrix (or 3 eigenvectors * 3)
)
# We will use 3 eigenvalues + 9 eigenvector components = 12, or simplified.
# Let's stick to: 3 eigenvalues + 9 eigenvector components = 12.
DROPOUT_RATE = 0.1

# Output
OUTPUT_DIM = 3  # Predicting vector components (nx, ny, nz)

# ---------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------
BATCH_SIZE = 256  # Adjust based on GPU VRAM (A100 40GB can handle large batches)
EPOCHS = 15  # Total training epochs
LEARNING_RATE = 1e-3  # Initial learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization
PATIENCE = 3  # Early stopping patience (epochs without improvement)
NUM_WORKERS = 2  # Number of dataloader workers

# ---------------------------------------------------------
# Reproducibility & Hardware
# ---------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Normalization Constants (Approximate from EDA)
# ---------------------------------------------------------
# Used to standardize inputs before feeding to the network
# These can be refined by calculating over the training set,
# but fixed constants ensure consistency during inference.
MEAN_TIME = 13000.0
STD_TIME = 5000.0
MEAN_CHARGE = 4.0
STD_CHARGE = 16.0
# Coordinates are roughly centered at 0 with range ~500m,
# but we usually normalize by a fixed scale factor (e.g., 500) or standard deviation.
COORD_SCALE = 500.0
