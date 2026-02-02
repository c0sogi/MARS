import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_58"
SUBMISSION_DIR = "./submission"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 12
FLOAT_PRECISION = "float64"  # Strictly float64 to minimize numerical noise

# =============================================================================
# DATASET & FEATURE CONFIGURATION
# =============================================================================
# Prefixes for the pre-extracted features in the CSV files
PREFIX_MARGIN = "margin"
PREFIX_SHAPE = "shape"
PREFIX_TEXTURE = "texture"

# Physical Feature Extraction Parameters
INVERT_THRESHOLD = 0.5  # Threshold for polarity correction (white background check)

# =============================================================================
# TRANSFORMER HYPERPARAMETERS
# =============================================================================
# Robust Topology
QUANTILE_N_QUANTILES = 50
QUANTILE_OUTPUT_DIST = "normal"

# Interaction Topology
INTERACTION_LDA_COMPONENTS = 12
POLY_DEGREE = 2

# LDA Estimator Parameters
LDA_SOLVER = "lsqr"
# Shrinkage candidates as per strategy: Fixed [0.001, 0.01]
LDA_SHRINKAGE_CANDIDATES = [0.001, 0.01]

# =============================================================================
# EXPERT LIBRARY DEFINITIONS
# =============================================================================
# Scopes
SCOPE_GLOBAL = "global"
SCOPE_MARGIN = "margin"
SCOPE_SHAPE = "shape"
SCOPE_TEXTURE = "texture"
SCOPE_PHYSICAL = "physical"

# Topologies
TOPOLOGY_MARGINAL = "marginal"  # PowerTransformer(yeo-johnson)
TOPOLOGY_ROTATIONAL = "rotational"  # Power -> PCA(whiten=False) -> Power
TOPOLOGY_ROBUST = "robust"  # QuantileTransformer(normal, n=50)
TOPOLOGY_INTERACTION = "interaction"  # Power -> LDA_Tr(n=12) -> Poly(2) -> Power
TOPOLOGY_PHYSICAL_POLY = "physical_poly"  # Poly(2) on Physical view

# Generate the list of expert configurations
EXPERT_DEFINITIONS = []

# 1. Global Scope Experts
# Topologies: Marginal, Rotational, Robust
for topo in [TOPOLOGY_MARGINAL, TOPOLOGY_ROTATIONAL, TOPOLOGY_ROBUST]:
    for shrink in LDA_SHRINKAGE_CANDIDATES:
        EXPERT_DEFINITIONS.append(
            {"scope": SCOPE_GLOBAL, "topology": topo, "shrinkage": shrink}
        )

# 2. Semantic Scope Experts (Margin, Shape, Texture)
# Topologies: Marginal, Rotational, Robust, Interaction
for scope in [SCOPE_MARGIN, SCOPE_SHAPE, SCOPE_TEXTURE]:
    for topo in [
        TOPOLOGY_MARGINAL,
        TOPOLOGY_ROTATIONAL,
        TOPOLOGY_ROBUST,
        TOPOLOGY_INTERACTION,
    ]:
        for shrink in LDA_SHRINKAGE_CANDIDATES:
            EXPERT_DEFINITIONS.append(
                {"scope": scope, "topology": topo, "shrinkage": shrink}
            )

# 3. Physical Scope Experts
# Topologies: Marginal (Baseline), Physical-Polynomial (Specific)
for topo in [TOPOLOGY_MARGINAL, TOPOLOGY_PHYSICAL_POLY]:
    for shrink in LDA_SHRINKAGE_CANDIDATES:
        EXPERT_DEFINITIONS.append(
            {"scope": SCOPE_PHYSICAL, "topology": topo, "shrinkage": shrink}
        )

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================
# File paths for caching intermediate physical features
CACHE_PHYSICAL_TRAIN = os.path.join(WORKING_DIR, "physical_features_train.parquet")
CACHE_PHYSICAL_VAL = os.path.join(WORKING_DIR, "physical_features_val.parquet")
CACHE_PHYSICAL_TEST = os.path.join(WORKING_DIR, "physical_features_test.parquet")

# Prefix for caching expert predictions during selection phase
CACHE_PREDICTIONS_PREFIX = os.path.join(WORKING_DIR, "preds_")
