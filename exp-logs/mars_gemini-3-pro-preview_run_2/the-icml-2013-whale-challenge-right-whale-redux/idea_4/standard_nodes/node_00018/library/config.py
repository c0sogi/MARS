import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata Paths
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for this specific idea (Idea 4)
# Used for caching processed data and saving models
WORKING_DIR = "./working/idea_4"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Audio Processing Parameters
# ==========================================
# Physics-aligned parameters for Right Whale calls
SAMPLE_RATE = 2000
N_FFT = 1024  # High frequency resolution (513 bins)
HOP_LENGTH = 64  # High temporal resolution (~32ms)
N_MELS = 128  # Dense spectral representation
FMIN = 0
FMAX = None  # Defaults to Nyquist (SAMPLE_RATE / 2)

# Image dimensions for the model input (Time, Frequency)
# Time steps = (2s * 2000) / 64 approx 62-63 frames.
# We can fix a size or let it be dynamic. EfficientNet usually takes square-ish inputs.
# However, for audio, we usually keep the computed spectrogram shape.
# EfficientNet-B0 default is 224x224, but can handle other sizes.

# ==========================================
# Model Configuration
# ==========================================
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1
USE_GEM_POOLING = True  # Generalized Mean Pooling
PRETRAINED = True
IN_CHANNELS = 1  # Spectrogram is 1 channel

# ==========================================
# Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 128  # Maximized for stability with WeightedRandomSampler
EPOCHS = 20  # Max epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # For AdamW
PATIENCE = 5  # Early stopping patience

# Scheduler settings (Cosine Annealing)
T_MAX = EPOCHS
ETA_MIN = 1e-6

# Hardware settings
NUM_WORKERS = 4  # 12 vCPUs available, 4 is usually a safe sweet spot per loader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Caching Configuration
# ==========================================
# Filenames for cached numpy arrays to speed up training
TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npy")
TRAIN_LABELS_CACHE = os.path.join(WORKING_DIR, "train_labels.npy")
VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npy")
VAL_LABELS_CACHE = os.path.join(WORKING_DIR, "val_labels.npy")
TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npy")
TEST_CLIPS_CACHE = os.path.join(WORKING_DIR, "test_clips.npy")
