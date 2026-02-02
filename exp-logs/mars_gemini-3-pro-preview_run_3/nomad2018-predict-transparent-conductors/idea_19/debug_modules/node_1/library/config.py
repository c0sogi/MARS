import os

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_19"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# Global Constants
# ==========================================
RANDOM_SEED = 42
ATOMIC_SPECIES = ["Al", "Ga", "In", "O"]
METALS = ["Al", "Ga", "In"]
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# ==========================================
# Feature Extraction Hyperparameters
# ==========================================
# Cutoff radius (Angstroms) for determining bonded neighbors (Coordination Number)
# Based on typical bond lengths in oxides (Al-O ~1.9A, In-O ~2.1A), 3.0A captures the first shell safely.
NEIGHBOR_CUTOFF = 3.0

# Cutoff radius (Angstroms) for Radial Distribution Functions (RDF)
RDF_CUTOFF = 6.0

# Number of bins for RDF histograms
RDF_BINS = 60

# Specific Coordination Numbers (CN) to fingerprint explicitly
# We track the fraction of atoms with these CNs for each metal species.
INTERESTING_CNS = [4, 5, 6]

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
# Base parameters for the XGBoost Regressor
# "Use a low learning rate (e.g., eta approx 0.01) and a high number of estimators (e.g., 3,000)"
# "Stochastic Subsampling: ... approx 0.6 - 0.7"
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient histogram-based algorithm
}


# ==========================================
# Configuration Helper Functions
# ==========================================
def get_xgb_params(debug=False):
    """
    Returns XGBoost hyperparameters.

    Args:
        debug (bool): If True, returns parameters suitable for a quick debug run
                      (fewer estimators, higher learning rate).

    Returns:
        dict: Dictionary of XGBoost parameters.
    """
    params = XGB_PARAMS.copy()
    if debug:
        # Faster training for debugging
        params["n_estimators"] = 50
        params["learning_rate"] = 0.1
        params["max_depth"] = 4
    return params


def get_dataset_size(debug=False):
    """
    Returns the number of samples to use.

    Args:
        debug (bool): If True, returns a small number for testing pipeline.

    Returns:
        int or None: Number of samples (None means all).
    """
    return 100 if debug else None
