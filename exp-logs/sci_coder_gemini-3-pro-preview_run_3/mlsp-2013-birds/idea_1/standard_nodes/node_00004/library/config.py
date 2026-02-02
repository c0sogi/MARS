import os

# ==========================================
# Path Configuration
# ==========================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Feature data paths
# The histogram of segments is provided in supplemental_data
HISTOGRAM_PATH = os.path.join(
    INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
)

# Output paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Model Architecture Configuration
# ==========================================

# Input dimension corresponds to the 100 clusters in the bag-of-words feature vector
INPUT_DIM = 100

# Hidden layer dimension for the shallow MLP
HIDDEN_DIM = 64

# Number of bird species to predict
NUM_CLASSES = 19

# Dropout rate for regularization
DROPOUT_RATE = 0.5

# ==========================================
# Training Configuration
# ==========================================

# Reproducibility
RANDOM_SEED = 42

# Optimization hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Early stopping patience (number of epochs with no improvement on validation metric)
EARLY_STOPPING_PATIENCE = 10
