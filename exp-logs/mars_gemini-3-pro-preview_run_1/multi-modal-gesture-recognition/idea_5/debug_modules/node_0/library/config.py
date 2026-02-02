import os

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
SEED = 42
NUM_CLASSES = 21  # 0: Background, 1-20: Gestures
NUM_JOINTS = 20
JOINT_CHANNELS = 3  # x, y, z coordinates
AUDIO_CHANNELS = 1
AUDIO_SAMPLERATE = 16000

# Audio Feature Extraction
# Video FPS is approx 20fps -> 50ms per frame
# 50ms * 16000Hz = 800 samples
MFCC_N_MFCC = 13
MFCC_HOP_LENGTH = 800
MFCC_N_FFT = 2048

# Label Mapping
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
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

# Skeleton Graph Structure (Adjacency List)
# Indices map to:
# 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
# 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
# 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight,
# 12:HipLeft, 13:KneeLeft, 14:AnkleLeft, 15:FootLeft,
# 16:HipRight, 17:KneeRight, 18:AnkleRight, 19:FootRight
SKELETON_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),  # Spine & Head
    (2, 4),
    (4, 5),
    (5, 6),
    (6, 7),  # Left Arm
    (2, 8),
    (8, 9),
    (9, 10),
    (10, 11),  # Right Arm
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),  # Left Leg
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),  # Right Leg
]

# ==========================================
# Model Hyperparameters
# ==========================================
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
USE_CONTEXT_GATING = True
USE_GRAPH_CONV = True

# ==========================================
# Training Hyperparameters
# ==========================================
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience
GRAD_CLIP = 1.0

# Loss Configuration
LABEL_SMOOTHING = 0.1
BG_CLASS_WEIGHT = 0.5  # Downweight background class to encourage recall

# Debugging / Development
DEBUG = False
DEBUG_SUBSET_SIZE = 20
