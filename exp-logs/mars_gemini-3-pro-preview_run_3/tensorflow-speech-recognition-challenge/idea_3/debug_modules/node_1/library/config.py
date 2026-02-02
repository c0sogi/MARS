import os

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

PATHS = {
    "train_csv": os.path.join(METADATA_DIR, "train.csv"),
    "val_csv": os.path.join(METADATA_DIR, "val.csv"),
    "test_csv": os.path.join(METADATA_DIR, "test.csv"),
    "train_audio_dir": os.path.join(INPUT_DIR, "train", "audio"),
    "test_audio_dir": os.path.join(INPUT_DIR, "test", "audio"),
    "model_save_path": os.path.join(WORKING_DIR, "best_model.pth"),
    "submission_path": os.path.join(SUBMISSION_DIR, "submission.csv"),
    "cache_dir": os.path.join(WORKING_DIR, "cache"),
}

# ==========================================
# Audio Configuration
# ==========================================
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 64,  # Matches ResNet capacity
    "n_fft": 480,  # ~30ms window (0.030 * 16000)
    "hop_length": 160,  # ~10ms overlap (0.010 * 16000)
    "f_min": 0,
    "f_max": None,  # Defaults to sample_rate // 2
    "duration": 1.0,  # 1 second clips
    "num_samples": 16000,  # sample_rate * duration
}

# ==========================================
# Model Configuration
# ==========================================
MODEL_CONFIG = {
    "num_classes": 12,
    "hidden_size": 128,  # Hidden dimension for BiGRU and Attention
    "backbone": "resnet34",  # Feature extractor
    "pretrained": True,  # Use ImageNet weights
    "dropout": 0.3,  # Dropout probability
    "use_attention": True,  # Enable Attention Pooling
}

# ==========================================
# Training Configuration
# ==========================================
TRAIN_CONFIG = {
    "batch_size": 128,  # Fits in A100 memory
    "learning_rate": 1e-3,  # Standard for AdamW
    "num_epochs": 25,  # Sufficient for convergence
    "seed": 42,  # Reproducibility
    "weight_decay": 1e-4,  # Regularization
    "early_stopping_patience": 6,
    "num_workers": 4,  # Data loading workers
    "debug": False,  # Set True to use a small subset of data
    "debug_samples": 1000,  # Number of samples if debug is True
}

# ==========================================
# Label Definitions
# ==========================================
LABELS = [
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
    "silence",
    "unknown",
]

# Mappings
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}
