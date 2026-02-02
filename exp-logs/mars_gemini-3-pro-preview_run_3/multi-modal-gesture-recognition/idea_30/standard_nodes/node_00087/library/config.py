import os

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_30")
SUBMISSION_DIR = "./submission"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BEST_MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Temporal Windowing
WINDOW_SIZE = 64
STRIDE = 32  # Overlap for training generation

# Feature Dimensions
NUM_CLASSES = 21  # 20 Gestures + 1 Background
NUM_JOINTS = 20
COORDS_PER_JOINT = 3  # X, Y, Z
DERIVATIVES = 3  # Position, Velocity, Acceleration
SKELETON_DIM = NUM_JOINTS * COORDS_PER_JOINT * DERIVATIVES  # 180 features

# Audio Features
AUDIO_MFCC_N_MFCC = 13
AUDIO_DIM = AUDIO_MFCC_N_MFCC

# Total Input Dimension for the Model
INPUT_DIM = SKELETON_DIM + AUDIO_DIM  # 193 features

# Scaling
# Convert mm to meters to align magnitude with MFCCs (Lesson 00082)
SKELETON_SCALE_FACTOR = 0.001

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Stage 1: Encoder
HIDDEN_DIM = 256  # Bi-GRU (128 units per direction * 2)
ENCODER_LAYERS = 1
DROPOUT = 0.3

# Stage 2 & 3: MS-TCN Refinement
# Monotonic Dilation Schedule (Lesson 00080)
STAGE_DILATIONS = [1, 2, 4, 8, 16]
KERNEL_SIZE = 3

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Function Weights
BG_CLASS_WEIGHT = 0.2  # Downweight background class (Lesson 00010)
SMOOTHING_LOSS_WEIGHT = 0.15
SMOOTHING_THRESHOLD = 1.0  # Truncated MSE threshold (Lesson 00055)

# ==========================================
# Inference & Post-Processing
# ==========================================
INFERENCE_OVERLAP_RATIO = 0.5  # 50% overlap for sliding window inference
MIN_GESTURE_LENGTH = 5  # Minimum frames to consider a valid gesture (Lesson 00071)

# ==========================================
# Debugging / Development
# ==========================================
# Set to True to run on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_SUBSET_SIZE = 10  # Number of samples to use if DEBUG is True
