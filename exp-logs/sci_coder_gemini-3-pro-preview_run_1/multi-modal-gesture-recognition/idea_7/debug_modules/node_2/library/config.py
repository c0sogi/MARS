import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
STATS_PATH = os.path.join(WORKING_DIR, "stats.npz")
CACHE_DIR = WORKING_DIR  # Directory for caching processed data

# ==========================================
# Data Configuration
# ==========================================
SEED = 42
NUM_CLASSES = 21  # 20 Gestures + 1 Background (Index 0)
BACKGROUND_LABEL = 0

# Label Mapping (Name -> ID)
LABEL_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}

# Reverse Mapping (ID -> Name)
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
ID_TO_NAME[BACKGROUND_LABEL] = "background"

# Feature Dimensions
# Skeleton: 20 joints * 3 (x,y,z) = 60
INPUT_DIM_POS = 60
INPUT_DIM_VEL = 60
# Audio: MFCC coefficients
INPUT_DIM_AUDIO = 13

# ==========================================
# Model Hyperparameters (KA-GRN)
# ==========================================
HIDDEN_DIM = 256
NUM_LAYERS = 2  # Bidirectional GRU layers
DROPOUT = 0.3
USE_BIDIRECTIONAL = True

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 8  # Micro-batching for small dataset
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05  # Aggressive regularization

# Loss Configuration
LAMBDA_BOUNDARY = 0.5  # Weight for auxiliary boundary loss
LABEL_SMOOTHING = 0.1
BACKGROUND_WEIGHT = 0.5  # Weight for background class in CE loss

# Inference Configuration
MEDIAN_FILTER_KERNEL = 5
MIN_GESTURE_LENGTH = 5
