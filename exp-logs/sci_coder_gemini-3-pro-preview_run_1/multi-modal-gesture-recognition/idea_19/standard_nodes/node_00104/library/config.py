import os

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Specific working directory for this idea iteration
WORKING_DIR = "./working/idea_19"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure mutable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Label Configuration
# ==========================================
# 0 is reserved for the background/null class
BACKGROUND_CLASS_ID = 0
NUM_GESTURE_CLASSES = 20
TOTAL_CLASSES = NUM_GESTURE_CLASSES + 1  # 0 to 20

# Mapping from Gesture Name to ID (1-20)
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

# Reverse Mapping
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
ID_TO_NAME[BACKGROUND_CLASS_ID] = "background"

# ==========================================
# Hyperparameters & Model Config
# ==========================================
HYPERPARAMS = {
    # General
    "seed": 42,
    "num_epochs": 65,
    "batch_size": 8,  # Micro-batching strategy
    "early_stopping_patience": 12,
    # Optimization
    "learning_rate": 1e-3,
    "weight_decay": 0.05,  # Aggressive regularization
    "scheduler_min_lr": 1e-6,
    # Loss Configuration
    "bg_weight": 0.35,  # Critical: Downweight background to prevent collapse (Cite {solution_lesson_node_00095})
    "label_smoothing": 0.1,  # Handle boundary ambiguity
    # Architecture
    "hidden_dim": 320,  # Increased capacity (Cite {solution_lesson_node_00061})
    "dropout": 0.3,
    "kernel_size_temporal": 7,  # Robust local receptive field
    # Audio Preprocessing (Physics-Based Alignment)
    "audio_sample_rate": 16000,
    "video_fps": 20,
    "n_mfcc": 13,  # Compact features to prevent noise overfitting
    "n_fft": 2048,  # Large window for temporal overlap
    "hop_length": 800,  # 16000 / 20 = 800 samples per frame alignment
    # Augmentation
    "resample_alpha_min": 0.8,
    "resample_alpha_max": 1.2,
    "mask_channel_prob": 0.1,
    "temporal_mask_prob": 0.5,  # (Cite {solution_lesson_node_00040})
    "temporal_mask_size_min": 0.05,
    "temporal_mask_size_max": 0.20,
    # Inference
    "median_filter_size": 5,
    "min_gesture_length": 5,
}
