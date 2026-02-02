import os

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 12  # Available vCPUs for parallel processing

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_60"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Ensure the working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================================================
# DATA FILE PATHS
# =============================================================================
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
# Column prefixes for the provided feature sets in the CSV files
FEATURE_PREFIXES = {"margin": "margin", "shape": "shape", "texture": "texture"}

# Image processing settings for Morphometrics extraction
# Threshold to determine if image needs inversion (if corner pixels are white/high intensity)
MORPH_INVERT_THRESHOLD = 0.5

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# LDA Shrinkage candidates for the ensemble selection phase.
# 'auto' typically corresponds to Ledoit-Wolf or OAS depending on the solver implementation.
LDA_SHRINKAGE_CANDIDATES = [0.001, 0.01, 0.1, "auto"]

# =============================================================================
# EXPERT GROUP CONFIGURATIONS
# =============================================================================

# Group A: Global Statistical Anchors
# Operates on the full 192-feature vector (Global View).
GROUP_A_CONFIG = {
    "name": "Global_Anchors",
    "feature_source": "global",
    "pipelines": [
        {"name": "Marginal", "steps": [("power", {"method": "yeo-johnson"})]},
        {
            "name": "Rotational",
            "steps": [
                ("power", {"method": "yeo-johnson"}),
                (
                    "pca",
                    {
                        "n_components": 0.99,
                        "whiten": False,
                        "random_state": RANDOM_SEED,
                    },
                ),
                ("power_2", {"method": "yeo-johnson"}),
            ],
        },
    ],
}

# Group B: Stratified Rotational Experts
# Splits features by type, rotates them independently to preserve manifold structure, then concatenates.
GROUP_B_CONFIG = {
    "name": "Stratified_Rotational",
    "feature_source": "stratified",
    "subsets": ["margin", "shape", "texture"],
    "subset_pipeline": [
        ("power", {"method": "yeo-johnson"}),
        ("pca", {"n_components": 0.99, "whiten": False, "random_state": RANDOM_SEED}),
        ("power_2", {"method": "yeo-johnson"}),
    ],
}

# Group C: Hierarchical Interaction Experts
# Level 1: Intra-Domain Interactions (Discriminative projection -> Polynomials)
GROUP_C_INTRA_CONFIG = {
    "name": "Intra_Domain_Interaction",
    "feature_source": "stratified",
    "subsets": ["margin", "shape", "texture"],
    "subset_pipeline": [
        ("power", {"method": "yeo-johnson"}),
        ("lda_transform", {"n_components": 10}),  # Supervised dimensionality reduction
        ("poly", {"degree": 2, "interaction_only": False, "include_bias": False}),
        ("power_2", {"method": "yeo-johnson"}),
    ],
}

# Level 2: Inter-Domain Interactions (Pairwise Discriminative projection -> Interaction Polynomials)
GROUP_C_INTER_CONFIG = {
    "name": "Inter_Domain_Interaction",
    "feature_source": "stratified_pairs",
    "pairs": [("margin", "shape"), ("margin", "texture"), ("shape", "texture")],
    "pre_concat_pipeline": [
        ("power", {"method": "yeo-johnson"}),
        ("lda_transform", {"n_components": 10}),
    ],
    "post_concat_pipeline": [
        ("poly", {"degree": 2, "interaction_only": True, "include_bias": False}),
        ("power", {"method": "yeo-johnson"}),
    ],
}

# Group D: Physical Polynomial Experts
# Operates on extracted Morphometrics (Hu Moments + Geometric Scalars) from binary images.
GROUP_D_CONFIG = {
    "name": "Physical_Polynomial",
    "feature_source": "morphometrics",
    "pipeline": [
        ("power", {"method": "yeo-johnson"}),
        ("poly", {"degree": 2, "interaction_only": False, "include_bias": False}),
        ("power_2", {"method": "yeo-johnson"}),
    ],
}
