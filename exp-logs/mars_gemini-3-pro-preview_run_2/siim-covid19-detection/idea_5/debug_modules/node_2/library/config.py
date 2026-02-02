import os
import torch
import random
import numpy as np

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# ==========================================
# General Configuration
# ==========================================
SEED = 42
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_WORKERS = 4
DEBUG = False
DEBUG_SAMPLE_SIZE = 100

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 800
BATCH_SIZE = 8  # Adjusted for A100 and ResNet101 memory footprint
PIXEL_MEAN = [0.485, 0.456, 0.406]  # ImageNet defaults
PIXEL_STD = [0.229, 0.224, 0.225]

# ==========================================
# Model Configuration
# ==========================================
BACKBONE = "resnet101"
# Detection Classes: 0=Background, 1=Typical, 2=Indeterminate, 3=Atypical
NUM_DETECTION_CLASSES = 4
# Study Classes: Negative, Typical, Indeterminate, Atypical
NUM_STUDY_CLASSES = 4

# RPN & ROI Heads Settings (High Capacity)
RPN_PRE_NMS_TOP_N_TRAIN = 4000
RPN_POST_NMS_TOP_N_TRAIN = 3000
RPN_PRE_NMS_TOP_N_TEST = 2000
RPN_POST_NMS_TOP_N_TEST = 1000
BOX_DETECTIONS_PER_IMG = 200
BOX_SCORE_THRESH = 0.05
BOX_NMS_THRESH = 0.5

# MIL Head Settings
MIL_POOL_SIZE = 64  # Top K proposals to pool for MIL

# ==========================================
# Training Configuration
# ==========================================
NUM_EPOCHS = 12
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_DECAY_STEP = 8
LR_GAMMA = 0.1
GRADIENT_CLIP_NORM = 10.0

# Loss Weights
MIL_LOSS_WEIGHT = 0.5

# ==========================================
# Label Mappings
# ==========================================
# Study Labels
STUDY_ID_TO_LABEL = {
    0: "Negative for Pneumonia",
    1: "Typical Appearance",
    2: "Indeterminate Appearance",
    3: "Atypical Appearance",
}
STUDY_LABEL_TO_ID = {v: k for k, v in STUDY_ID_TO_LABEL.items()}

# Detection Labels (Internal Training)
# Note: 0 is reserved for background in Faster R-CNN
DETECTION_ID_TO_LABEL = {
    1: "Typical Appearance",
    2: "Indeterminate Appearance",
    3: "Atypical Appearance",
}
DETECTION_LABEL_TO_ID = {v: k for k, v in DETECTION_ID_TO_LABEL.items()}

# Submission Constants
SUBMISSION_DETECTION_LABEL = "opacity"
NONE_PREDICTION = "none 1 0 0 1 1"


# ==========================================
# Utility Functions
# ==========================================
def seed_everything(seed=SEED):
    """
    Seeds all random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
