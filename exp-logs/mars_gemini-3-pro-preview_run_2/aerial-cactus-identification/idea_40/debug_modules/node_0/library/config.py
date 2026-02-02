import os
import torch

# ==========================================
#         Directory & File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_40"

# Ensure working directories exist immediately upon import
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "submission"), exist_ok=True)
os.makedirs(os.path.join(WORKING_DIR, "cache"), exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")

# ==========================================
#         Model Hyperparameters
# ==========================================
# Ultra-Wide RepResNeXt Configuration
MODEL_NAME = "UltraWide_SE_RepResNeXt"
IMAGE_SIZE = (32, 32)
IN_CHANNELS = 3
NUM_CLASSES = 1

# Backbone Architecture
# Scaling width to [96, 192, 384] as per "Ultra-Wide" specification
STAGES_CHANNELS = [96, 192, 384]
CARDINALITY = 32  # Group size for RepNeXt blocks
USE_SE = True  # Squeeze-and-Excitation

# ==========================================
#         Training Hyperparameters
# ==========================================
# Homogeneous Seed Averaging
SEED_LIST = [0, 1, 2, 3, 4]

# Optimization
BATCH_SIZE = 128  # A100 can handle large batches, 128 is safe/stable
EPOCHS = 8  # Aggressively reduced schedule (5-8 epochs)
LEARNING_RATE = 1e-3  # Standard for AdamW
WEIGHT_DECAY = 1e-2
COSINE_T_MAX = EPOCHS  # For Cosine Annealing Scheduler

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # 12 vCPUs available

# ==========================================
#         Debugging / Development
# ==========================================
# Flags to control dataset size for debugging
DEBUG = False
DEBUG_SAMPLE_SIZE = 100
