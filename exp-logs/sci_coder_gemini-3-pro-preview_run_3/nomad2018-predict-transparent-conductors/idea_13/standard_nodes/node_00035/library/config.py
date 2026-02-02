import os

# ==========================================
# 1. File System Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Cached Feature Files (Parquet)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# ==========================================
# 2. Global Constants
# ==========================================
RANDOM_SEED = 42
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
LOG_TRANSFORM_TARGETS = True

# Atomic Groupings for Aggregation
METALS = ["Al", "Ga", "In"]
ANIONS = ["O"]

# ==========================================
# 3. Structural Feature Hyperparameters
# ==========================================

# Radial Distribution Function (RDF) Settings
# Captures pairwise distances.
RDF_CUTOFF = 6.0  # Angstroms
RDF_BINS = 40
RDF_SIGMA = 0.2  # Width of Gaussian smearing (if used) or bin width implicitly
# Specific pairs to compute RDFs for (Element A, Element B)
# We focus on Metal-Anion and Metal-Metal interactions
RDF_PAIRS = [
    ("Al", "O"),
    ("Ga", "O"),
    ("In", "O"),  # Metal-Oxygen
    ("Al", "Al"),
    ("Ga", "Ga"),
    ("In", "In"),  # Homo-Metal
    ("Al", "Ga"),
    ("Al", "In"),
    ("Ga", "In"),  # Hetero-Metal
    ("O", "O"),  # Anion-Anion
]

# Angular Distribution Function (ADF) Settings
# Captures 3-body bond angles.
# We use a shorter cutoff to define "bonds" for angle calculation.
ADF_CUTOFF = 3.0  # Angstroms (Bond cutoff)
ADF_BINS = 30  # Number of bins for 0-180 degrees
# We aggregate angles into chemically meaningful categories rather than specific triplets
# Format: (Center_Type, Neighbor_Type_1, Neighbor_Type_2)
# 'Metal' matches any of Al, Ga, In. 'O' matches Oxygen.
ADF_TRIPLETS = [
    # Polyhedral angles: O - Metal - O (Internal polyhedron structure)
    ("Metal", "O", "O"),
    # Bridging angles: Metal - O - Metal (Connectivity between polyhedra)
    ("O", "Metal", "Metal"),
]

# ==========================================
# 4. Model Hyperparameters (XGBoost)
# ==========================================
# These parameters are chosen for robustness on tabular/structural data
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 7,
    "subsample": 0.65,
    "colsample_bytree": 0.65,
    "min_child_weight": 1,
    "gamma": 0.1,  # Minimum loss reduction
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Faster training
}

# Training loop settings
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 500


# ==========================================
# 5. Utility Functions
# ==========================================
def setup_directories():
    """Ensures that working and submission directories exist."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print(f"Directories ensured: {WORKING_DIR}, {SUBMISSION_DIR}")
