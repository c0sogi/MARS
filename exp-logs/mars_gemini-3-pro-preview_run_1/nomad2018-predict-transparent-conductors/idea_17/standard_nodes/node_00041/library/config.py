import os
import torch

# =============================================================================
# Global Configuration
# =============================================================================

# Random Seed for Reproducibility
SEED = 42

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# File Paths
# =============================================================================

# Input Directories (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working Directory (Read/Write for Cache and Models)
# Using a specific subdirectory for this idea to avoid conflicts
WORKING_DIR = "./working/idea_17"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Files for Preprocessed Data
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")
SCALERS_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.npz")

# Model Checkpoint Path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")

# =============================================================================
# Hyperparameters
# =============================================================================

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
PATIENCE = 20  # Early stopping patience
WEIGHT_DECAY = 1e-4  # Regularization for wide layers

# Model Architecture
# Atomic Stream (Wide Point Processor)
ATOMIC_INPUT_DIM = 9  # 4 (One-hot) + 3 (Coords) + 1 (NN Dist) + 1 (Potential)
ATOMIC_HIDDEN_DIM = 512  # Wide MLP
ATOMIC_DROPOUT = 0.2

# Global Stream (Thermodynamic Context)
GLOBAL_INPUT_DIM = 12  # 3 (Lattice Vecs) + 3 (Angles) + 1 (Vol) + 1 (Density) + 3 (Stoich) + 1 (Total Atoms)
GLOBAL_HIDDEN_DIM = 256
GLOBAL_DROPOUT = 0.2

# Fusion Head
FUSION_HIDDEN_DIM = 256
OUTPUT_DIM = 2  # Formation Energy, Bandgap Energy

# =============================================================================
# Feature Definitions
# =============================================================================

# Target Columns
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Atomic Identity Mapping
ATOM_TYPES = ["Al", "Ga", "In", "O"]
ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_TYPES)}

# Global Feature Columns (from CSV)
GLOBAL_FEATURE_COLS = [
    "lattice_vector_1_ang",
    "lattice_vector_2_ang",
    "lattice_vector_3_ang",
    "lattice_angle_alpha_degree",
    "lattice_angle_beta_degree",
    "lattice_angle_gamma_degree",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "number_of_total_atoms",
    # Derived features like volume and density are calculated dynamically
]
