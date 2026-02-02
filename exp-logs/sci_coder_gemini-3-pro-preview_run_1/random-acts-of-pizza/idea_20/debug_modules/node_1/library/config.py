import os

# ==========================================
# Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_20"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Global Settings
# ==========================================
RANDOM_STATE = 42
DEBUG = False  # Set to True to use a smaller subset of data for debugging

# ==========================================
# Feature Engineering Configuration
# ==========================================
# SBERT model for generating embeddings
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_EMBEDDING_DIM = 384

# TF-IDF Configuration for Random Forest
TFIDF_VOCAB_SIZE = 5000

# PCA Configuration for Latent Semantic Centroids
PCA_COMPONENTS = 20

# ==========================================
# Model Hyperparameters
# ==========================================

# --- Stream A: Random Forest ---
RF_ESTIMATORS = 500
RF_MAX_DEPTH = None  # Allow full depth
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1

# --- Stream B: MLP (Credibility-Gated Attention) ---
MLP_HIDDEN_DIM = 128
MLP_DROPOUT_RATE = 0.3
MLP_ATTENTION_HEADS = 4  # If using MultiheadAttention, though dot-product is specified
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 1e-4

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 15  # Early stopping patience

# ==========================================
# Ensemble Configuration
# ==========================================
ENSEMBLE_WEIGHT_RF = 0.5
ENSEMBLE_WEIGHT_MLP = 0.5
