import os
import torch

# ==========================================
# Global Configuration & Hyperparameters
# ==========================================

# 1. Random Seed & Reproducibility
SEED = 42
# Disable strict CuDNN determinism to maximize kernel performance (Cite Lesson 00070)
CUDNN_DETERMINISTIC = False

# 2. Directories
# Input directory (Read-Only)
INPUT_DIR = "./input"
# Metadata directory containing Parquet files
METADATA_DIR = "./metadata"

# Working directory for artifacts (cache, models) - Specific to Idea 37
WORKING_DIR = "./working/idea_37"
# Submission directory
SUBMISSION_DIR = "./submission"

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 3. File Paths
# Metadata Parquet files (Pre-split 80/20 Stratified)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Sample submission for format reference
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output paths
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
CACHE_DIR = WORKING_DIR  # Directory to store processed numpy/parquet cache

# 4. Training Hyperparameters
BATCH_SIZE = 4096
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2  # Standard for AdamW (Decoupled Weight Decay)
PATIENCE = 10  # Early Stopping Patience (Cite Lesson 00003)
SCHEDULER_FACTOR = 0.1  # Aggressive decay factor (Cite Lesson 00068)
SCHEDULER_PATIENCE = 3  # Patience for scheduler (usually < ES patience)

# Debugging parameter (Set to an integer to subsample data, None for full training)
DEBUG_SAMPLE_SIZE = None

# 5. Model Architecture Hyperparameters
# Deep Parallel Vector-DCN-ResNet (5-Block Scaled)
RESNET_BLOCKS = 5  # Scaled backbone (Cite Lesson 00054)
DCN_LAYERS = 3  # Asymmetric branch, decoupled from backbone (Cite Lesson 00071)
HIDDEN_DIM = 512  # Capacity (Cite Lesson 00029)
DROPOUT = 0.2  # Regularization (Cite Lesson 00056)
NOISE_STD = 0.01  # Input Gaussian Noise for continuous features (Cite Lesson 00053)

# 6. Feature Engineering Configuration
# Augmented Physics-Informed Engineering
USE_ASPECT_TRIG = True  # Calculate Sin/Cos but keep raw Aspect (Cite Lesson 00034)
USE_HYDRO_DIST = True  # Euclidean Distance to Hydrology (Cite Lesson 00016)
USE_HYDRO_ELEV = True  # Absolute Hydrology Elevation (Cite Lesson 00019)
USE_AMENITIES_MEAN = True  # Mean Distance to Amenities (Cite Lesson 00009)

# 7. Hardware Configuration
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
