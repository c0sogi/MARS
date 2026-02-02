import os
import torch

# ==========================================
# Global Configuration & Paths
# ==========================================
RANDOM_STATE = 42
N_THREADS = 12  # Matches vCPU count

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")

# Working Directory for Idea 10 (Neuro-Symbolic Stratified Ensemble)
WORKING_DIR = "./working/idea_10"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Input Files
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
DIPOLE_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
POTENTIAL_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
MULLIKEN_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
MAGNETIC_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
CONTRIB_CSV = os.path.join(INPUT_DIR, "scalar_coupling_contributions.csv")

# ==========================================
# Physics & Chemistry Constants
# ==========================================
# Atomic Numbers
ATOM_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

# Covalent Radii (Angstroms) - Used for Adaptive Graph Construction
# Source: Alvarez (2013) or standard Cordero et al.
COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}  # H  # C  # N  # O  # F

# Coupling Types
COUPLING_TYPES = ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]

# Graph Construction Thresholds
# Connectivity defined if dist < r_i + r_j + TOLERANCE
CONNECTIVITY_TOLERANCE = 0.3  # Angstroms

# ==========================================
# Stage 1: Graph Neural Network (MPNN) Config
# ==========================================
GNN_PARAMS = {
    "node_dim": 64,  # Dimension of atom embeddings
    "edge_dim": 128,  # Dimension of edge embeddings (RBF expansion)
    "hidden_dim": 256,  # Hidden dimension of interaction layers
    "output_dim": 128,  # Dimension of the learned interaction embedding to extract
    "num_layers": 4,  # Depth of the GNN
    "num_rbf": 128,  # Number of RBF kernels for distance encoding
    "cutoff": 10.0,  # Cutoff distance for RBF
    "aggr": "add",  # Aggregation scheme
    "dropout": 0.1,
    "batch_size": 128,  # Batch size for GNN training
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 25,  # Epochs for representation learning
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==========================================
# Stage 2: Stratified XGBoost Config
# ==========================================
# Base parameters for High-Capacity Boosting
XGB_PARAMS = {
    "objective": "reg:absoluteerror",  # Optimizing MAE directly
    "eval_metric": "mae",
    "tree_method": "gpu_hist",  # Use A100 GPU
    "booster": "gbtree",
    "learning_rate": 0.01,  # Low LR for high capacity
    "max_depth": 11,  # Deep trees (10-12 range)
    "subsample": 0.8,
    "colsample_bytree": 0.4,  # Aggressive feature subsampling for hybrid features
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 20000,  # High capacity, relying on early stopping
    "n_jobs": N_THREADS,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

# Early Stopping Rounds
XGB_EARLY_STOPPING_ROUNDS = 100

# ==========================================
# Feature Engineering & Pruning Config
# ==========================================
# Variance threshold for pruning features within a specific stratum (coupling type)
# Features with variance lower than this will be dropped for that specific model
PRUNING_VARIANCE_THRESHOLD = 1e-9


# ==========================================
# Helper Functions
# ==========================================
def setup_directories():
    """Creates necessary working directories."""
    dirs = [WORKING_DIR, CACHE_DIR, MODEL_DIR, SUBMISSION_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Directories initialized at {WORKING_DIR}")


def get_device():
    """Returns the torch device."""
    return torch.device(GNN_PARAMS["device"])
