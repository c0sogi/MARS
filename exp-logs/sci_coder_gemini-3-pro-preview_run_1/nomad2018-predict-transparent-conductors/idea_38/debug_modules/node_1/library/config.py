import os

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_38"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Constants
# ==========================================
# Atomic species present in the dataset
ATOM_TYPES = ["Al", "Ga", "In", "O"]

# Generate all unique pairs for structural statistics (including self-pairs)
# Order: (Al, Al), (Al, Ga), ..., (O, O)
PAIR_TYPES = []
for i in range(len(ATOM_TYPES)):
    for j in range(i, len(ATOM_TYPES)):
        PAIR_TYPES.append((ATOM_TYPES[i], ATOM_TYPES[j]))

# ==========================================
# Feature Dimensions
# ==========================================
# Atomic Stream Features:
# 1. One-hot encoding of atom type (4 dims)
# 2. Centered Cartesian coordinates x, y, z (3 dims)
# 3. Nearest Neighbor Distance (1 dim)
ATOMIC_FEATURE_DIM = 4 + 3 + 1  # Total: 8

# Global Stream Features:
# 1. Lattice vector lengths (3 dims)
# 2. Lattice angles (3 dims)
# 3. Unit cell volume (1 dim)
# 4. Atomic density (1 dim)
# 5. Total number of atoms (1 dim)
# 6. Composition percentages for Al, Ga, In (3 dims)
# 7. Mean Pairwise Distances for all 10 pair types (10 dims)
GLOBAL_FEATURE_DIM = 3 + 3 + 1 + 1 + 1 + 3 + len(PAIR_TYPES)  # Total: 22

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Atomic Stream Encoder (Wide MLP)
HIDDEN_DIM_ATOMIC = 512

# Global Stream Encoder (High-Capacity MLP)
HIDDEN_DIM_GLOBAL = 256

# Fusion Head
FUSION_HIDDEN_DIM = 256

# Regularization
DROPOUT_RATE = 0.1

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Regularization for wide layers
EPOCHS = 200
PATIENCE = 20  # Early stopping patience

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Debugging
# ==========================================
# Set to an integer (e.g., 100) to train on a small subset for debugging.
# Set to None for full training.
DEBUG_SAMPLE_SIZE = None
