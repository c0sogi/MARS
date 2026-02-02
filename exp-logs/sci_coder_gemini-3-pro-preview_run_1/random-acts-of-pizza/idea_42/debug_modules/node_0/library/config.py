import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
# Root directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_42")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Data Files (Generated Metadata)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Original Raw Data (for reference if needed, though metadata is preferred)
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
# Random Seed for reproducibility
SEED = 42

# SBERT Model for Semantic Embeddings
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2

# TF-IDF Configuration (Stream A)
TFIDF_VOCAB_SIZE = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Top-K Subreddits Configuration (Stream A)
TOP_K_SUBREDDITS = 50

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# --- Stream A: Consistency-Augmented Top-K Random Forest ---
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 1
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1
RF_RANDOM_STATE = SEED

# --- Stream B: Persona-Aware Skip-Gated MLP ---
# Architecture
MLP_HIDDEN_DIMS = [256, 128]  # Hidden layers for the dense branches
MLP_DROPOUT_EMB = 0.5  # Dropout applied to raw embeddings
MLP_DROPOUT_DENSE = 0.2  # Dropout applied to dense layers

# Training
MLP_BATCH_SIZE = 32
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_EPOCHS = 50
MLP_PATIENCE = 15  # Early stopping patience
MLP_OPTIMIZER = "AdamW"

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Weights for the simple weighted average ensemble
# Final Score = WEIGHT_RF * RF_Prob + WEIGHT_MLP * MLP_Prob
WEIGHT_RF = 0.5
WEIGHT_MLP = 0.5
