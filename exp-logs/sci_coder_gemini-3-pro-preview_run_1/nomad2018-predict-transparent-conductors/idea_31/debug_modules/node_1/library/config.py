import os
import torch

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Compute Device
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# File Paths
# ==========================================
# Input directories (Read-Only)
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_GEOMETRY_DIR = os.path.join(INPUT_DIR, "train")
TEST_GEOMETRY_DIR = os.path.join(INPUT_DIR, "test")

# Metadata directories (Generated previously)
METADATA_DIR = "./metadata"
METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
METADATA_VAL = os.path.join(METADATA_DIR, "val.csv")
METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching processed data
WORKING_DIR = "./working/idea_31"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Atomic Properties
# ==========================================
# Mapping element symbol -> [Atomic Mass, Covalent Radius, Electronegativity]
# Values based on standard periodic table data
ATOMIC_PROPERTIES = {
    "Al": [26.9815, 1.21, 1.61],
    "Ga": [69.7230, 1.22, 1.81],
    "In": [114.818, 1.42, 1.78],
    "O": [15.9990, 0.66, 3.44],
}

# ==========================================
# Data Preprocessing Parameters
# ==========================================
NEIGHBORS_K = 12  # Number of nearest neighbors for LCE calculation

# Feature Dimensions
# Atomic features: 4 (One-Hot) + 3 (Coords) + 1 (NN Dist) + 3 (LCE: Mass, Radius, Neg)
ATOMIC_FEATURE_DIM = 11
# Global features: 3 (Lattice Lens) + 3 (Angles) + 1 (Vol) + 1 (Density) + 3 (Stoich) + 1 (Total Atoms)
GLOBAL_FEATURE_DIM = 12

# ==========================================
# Model Hyperparameters
# ==========================================
MODEL_PARAMS = {
    "atomic_input_dim": ATOMIC_FEATURE_DIM,
    "global_input_dim": GLOBAL_FEATURE_DIM,
    "atomic_hidden_dim": 512,  # Wide MLP for atomic stream
    "global_hidden_dim": 256,  # Capacity for global stream
    "fusion_hidden_dim": 256,  # Dimension after concatenation
    "output_dim": 2,  # Formation energy and Bandgap energy
    "dropout": 0.1,  # Regularization
    "activation": "relu",  # Activation function
}

# ==========================================
# Training Hyperparameters
# ==========================================
TRAINING_PARAMS = {
    "batch_size": 64,
    "learning_rate": 5e-4,  # Slightly lower LR for stability with wide layers
    "weight_decay": 1e-4,  # L2 Regularization
    "epochs": 200,  # Maximum training epochs
    "patience": 20,  # Early stopping patience
    "factor": 0.5,  # LR Scheduler reduction factor
    "min_lr": 1e-6,  # Minimum learning rate
}
