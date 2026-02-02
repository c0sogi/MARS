import os
import torch
import random
import numpy as np

# ==========================================
# Reproducibility
# ==========================================
SEED = 42


def set_seed(seed=SEED):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Paths (via Metadata)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Path
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths (for deterministic processing)
# Used to store preprocessed features and encoders to avoid re-computation
TFIDF_VECTORIZER_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
MLB_PATH = os.path.join(WORKING_DIR, "mlb.joblib")

TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npz")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npz")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npz")

TRAIN_LABELS_PATH = os.path.join(WORKING_DIR, "train_labels.npz")
VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npz")

MODEL_PATH = os.path.join(WORKING_DIR, "sparse_mlp_model.pth")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
MAX_FEATURES = 25000  # Vocabulary size for TF-IDF (Input Dimension)
NGRAM_RANGE = (1, 2)  # Use Unigrams and Bigrams
MIN_DF = 5  # Minimum document frequency to filter rare terms
TOP_K_TAGS = 5000  # Number of most frequent tags to predict (Output Dimension)

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
HIDDEN_DIM = 1024  # Dimension of the hidden layer in the MLP
DROPOUT = 0.2  # Dropout rate for regularization

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 2048  # Large batch size to leverage A100 GPU memory and speed
LEARNING_RATE = 1e-3  # Initial learning rate for Adam optimizer
EPOCHS = 10  # Maximum number of training epochs
EARLY_STOPPING_PATIENCE = (
    3  # Stop training if validation F1 doesn't improve for 3 epochs
)
THRESHOLD = 0.35  # Probability threshold for converting logits to tags

# ==========================================
# Hardware Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Number of CPU cores available for data loading
