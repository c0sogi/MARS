import os

# -----------------------------------------------------------------------------
# File System Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
# Cache directory for deterministic data processing artifacts
CACHE_DIR = os.path.join(WORKING_DIR, "idea_27")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Global Constants & Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# Debugging control
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG_MODE is True

# -----------------------------------------------------------------------------
# Domain Knowledge: Chemical Properties
# -----------------------------------------------------------------------------
# Dictionary mapping chemical elements to their physical properties.
# Used for Anion Chemo-Structural Fingerprints.
# Electronegativity (EN) on Pauling scale.
# Ionic Radius in Angstroms (approximate for octahedral Al/Ga/In and O).
ELEMENT_PROPERTIES = {
    "Al": {"EN": 1.61, "Radius": 0.54},
    "Ga": {"EN": 1.81, "Radius": 0.62},
    "In": {"EN": 1.78, "Radius": 0.80},
    "O": {"EN": 3.44, "Radius": 1.40},
}

# -----------------------------------------------------------------------------
# Feature Extraction Configuration
# -----------------------------------------------------------------------------
FEATURE_CONFIG = {
    # Radial Distribution Function (RDF) parameters
    "rdf_cutoff": 6.0,  # Angstroms
    "rdf_bins": 30,  # Resolution of the histogram
    # Sublattice definitions
    "cation_elements": ["Al", "Ga", "In"],
    "anion_elements": ["O"],
    # Distributional Aggregation: Percentiles to compute for local descriptors
    # Captures the min, lower quartile, median, upper quartile, and max
    "percentiles": [0, 25, 50, 75, 100],
}

# -----------------------------------------------------------------------------
# Model Hyperparameters (XGBoost)
# -----------------------------------------------------------------------------
# Configuration for the XGBoost Regressor.
# Low learning rate and high estimators for robust generalization.
# Subsampling to prevent overfitting on specific structural features.
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Efficient histogram-based algorithm
}

# -----------------------------------------------------------------------------
# Training Pipeline Configuration
# -----------------------------------------------------------------------------
TRAIN_CONFIG = {
    "val_size": 0.2,
    "early_stopping_rounds": 100,
    "verbose_eval": 200,
    "target_cols": ["formation_energy_ev_natom", "bandgap_energy_ev"],
}
