import os
import random
import numpy as np
import torch

# -----------------------------------------------------------------------------
# 1. File Paths and Directories
# -----------------------------------------------------------------------------
INPUT_ROOT = "./input"
TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

# Metadata paths (pre-generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching and artifacts
WORKING_DIR = "./working/idea_15"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# 2. Audio Processing Parameters
# -----------------------------------------------------------------------------
# Configuration for generating High-Fidelity Log-Mel Spectrograms
AUDIO_PARAMS = {
    "sample_rate": 16000,
    "duration": 1.0,  # Fixed duration in seconds
    "n_fft": 1024,  # Window size (64ms)
    "hop_length": 160,  # Stride (10ms)
    "n_mels": 128,  # Number of Mel bands
    "f_min": 0,
    "f_max": 8000,  # Nyquist frequency
    "top_db": 80.0,  # For log conversion
}

# -----------------------------------------------------------------------------
# 3. Label Definitions & Mappings
# -----------------------------------------------------------------------------
# The 10 specific target commands required for the competition
TARGET_LABELS_SET = {
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
}

# The complete list of 31 fine-grained classes (30 words + silence)
# This allows the model to learn distinct features for auxiliary words
# instead of lumping them into a single 'unknown' class during training.
FINE_GRAINED_LABELS = sorted(
    [
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "house",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "wow",
        "yes",
        "zero",
        "silence",
    ]
)

# Mapping from label string to integer index
LABEL2ID = {label: i for i, label in enumerate(FINE_GRAINED_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def get_fine_grained_label_from_path(filepath):
    """
    Extracts the fine-grained label from the file path.
    Example: 'train/audio/bed/001.wav' -> 'bed'
             'train/audio/_background_noise_/white.wav' -> 'silence'
    """
    parts = filepath.split(os.sep)
    # The folder name is the second to last element
    folder = parts[-2] if len(parts) > 1 else "unknown"

    if folder == "_background_noise_":
        return "silence"
    return folder


def get_competition_label(fine_label):
    """
    Maps a fine-grained label to the 12 competition classes.
    Classes: yes, no, up, down, left, right, on, off, stop, go, silence, unknown.
    """
    if fine_label in TARGET_LABELS_SET:
        return fine_label
    if fine_label == "silence":
        return "silence"
    return "unknown"


# -----------------------------------------------------------------------------
# 4. Model Hyperparameters
# -----------------------------------------------------------------------------
MODEL_PARAMS = {
    "model_name": "efficientnet_b2",
    "num_classes": len(FINE_GRAINED_LABELS),  # 31
    "in_channels": 1,  # Spectrogram input is 1 channel
    "dropout": 0.3,
    "pretrained": True,
    "use_dilated_conv": True,  # Use dilated convolutions in the final stage
    "dilation_rate": 2,
}

# -----------------------------------------------------------------------------
# 5. Training Hyperparameters
# -----------------------------------------------------------------------------
TRAINING_PARAMS = {
    "seed": 42,
    "batch_size": 128,  # Optimized for A100 GPU
    "num_workers": 8,  # Utilizing available vCPUs
    "epochs": 30,  # Total training epochs
    # Optimization
    "learning_rate": 1e-3,  # Max LR for Cosine Scheduler
    "min_lr": 1e-5,  # Min LR for Cosine Scheduler
    "weight_decay": 1e-2,
    "swa_start_epoch": 25,
    "swa_lr": 1e-3,
    # Augmentation
    "mixup_alpha": 1.0,
    "spec_augment_prob": 0.5,
    "noise_injection_prob": 0.5,
    "noise_snr_min": 10,
    "noise_snr_max": 30,
    # Data Balancing
    "target_sample_count": 2000,  # Upsample target classes to this count
}


# -----------------------------------------------------------------------------
# 6. Utilities
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
