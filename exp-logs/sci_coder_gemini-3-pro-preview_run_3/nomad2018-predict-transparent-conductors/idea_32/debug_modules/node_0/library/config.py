import os
import random
import numpy as np

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory specific to this idea for caching intermediate results
WORKING_DIR = "./working/idea_32"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# ==========================================
# Reproducibility
# ==========================================
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for Python's random module and NumPy
    to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================
# Data Processing Configuration
# ==========================================
# Target variables to predict
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Bond Valence Parameters (R0 in Angstroms)
# Based on Brown & Altermatt (1985) for Metal(III) - Oxygen(-II) pairs
BVS_PARAMS = {
    "Al": 1.651,
    "Ga": 1.742,
    "In": 1.907,
    "b": 0.37,  # Universal softness parameter
}

# Feature Extraction Settings
RDF_MAX_R = 8.0  # Maximum radius for Radial Distribution Function (Angstroms)
RDF_BINS = 80  # Number of bins for RDF histogram
ANGLE_CUTOFF = 3.0  # Cutoff distance for bond angle calculation (Angstroms)

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
# Optimized for generalization with low learning rate and stochastic subsampling
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient histogram-based algorithm
}
