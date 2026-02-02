import os

# ==========================================
# Global Directories
# ==========================================
INPUT_DIR = "./metadata"
WORK_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data File Paths
# ==========================================
TRAIN_FILE = os.path.join(INPUT_DIR, "train.csv")
VAL_FILE = os.path.join(INPUT_DIR, "val.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Caching Paths (Idea 3 Specific)
# ==========================================
# We use .npy for numpy arrays and .bin for FAISS indices/models
# FastText model path (trained on corpus or loaded)
FASTTEXT_MODEL_PATH = os.path.join(WORK_DIR, "fasttext_custom.model")

# Processed Training Data (Vectors and Labels)
TRAIN_VECTORS_PATH = os.path.join(WORK_DIR, "train_vectors.npy")
TRAIN_LABELS_PATH = os.path.join(WORK_DIR, "train_labels.npy")
TRAIN_TOKENS_PATH = os.path.join(
    WORK_DIR, "train_tokens.npy"
)  # Optional: store raw tokens if needed for debugging

# FAISS Index Path
FAISS_INDEX_PATH = os.path.join(WORK_DIR, "knn_index.bin")

# Label Encoder Path (mapping class strings to integers)
LABEL_ENCODER_PATH = os.path.join(WORK_DIR, "label_encoder_classes.npy")

# ==========================================
# Hyperparameters
# ==========================================
SEED = 42

# k-NN Parameters
K_NEIGHBORS = 11  # Number of neighbors to retrieve
METRIC = "l2"  # Distance metric for FAISS (l2 or cosine)

# Data Processing Parameters
# Downsample 'PLAIN' class to this ratio to save memory and balance the index
PLAIN_SAMPLE_RATIO = 0.01

# FastText / Embedding Parameters
EMBEDDING_DIM = 100
CONTEXT_WINDOW = 2  # Number of tokens to look at left and right
MIN_COUNT = 1  # Minimum frequency for a word to be in FastText vocab
EPOCHS = 5  # Training epochs for the custom FastText model

# ==========================================
# Semiotic Classes
# ==========================================
# List of classes found in the dataset (based on EDA)
CLASSES = [
    "PLAIN",
    "PUNCT",
    "DATE",
    "LETTERS",
    "CARDINAL",
    "VERBATIM",
    "MEASURE",
    "ORDINAL",
    "DECIMAL",
    "MONEY",
    "DIGIT",
    "ELECTRONIC",
    "TELEPHONE",
    "TIME",
    "ADDRESS",
]
