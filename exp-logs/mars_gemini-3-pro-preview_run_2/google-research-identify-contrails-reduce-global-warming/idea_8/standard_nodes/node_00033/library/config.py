import os
import torch

# ==========================================
# Directory Configuration
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"

# Working directory for Idea 8 (Dilated ResNet18 U-Net)
WORKING_DIR = "./working/idea_8"
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# ==========================================
# Global Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Image & Data Configuration
# ==========================================
IMG_SIZE = 256
N_CHANNELS = 6  # 3 for Ash Composite + 3 for Temporal Difference

# Ash False Color Composite Normalization Bounds
# Based on standard GOES-16 Ash RGB recipe adapted for contrails
# Red: T15 - T14
# Green: T14 - T11
# Blue: T14
ASH_BOUNDS = {
    "T15_T14_MIN": -4.0,
    "T15_T14_MAX": 5.0,
    "T14_T11_MIN": -4.0,
    "T14_T11_MAX": 2.0,
    "T14_MIN": 243.0,
    "T14_MAX": 303.0,
}

# Band Indices (0-based from the file list 08-16)
# band_08 is index 0, band_11 is index 3, band_14 is index 6, band_15 is index 7
BAND_11_IDX = 3
BAND_14_IDX = 6
BAND_15_IDX = 7

# ==========================================
# Model Configuration
# ==========================================
MODEL_CONFIG = {
    "encoder_name": "resnet18",
    "encoder_weights": "imagenet",
    "in_channels": N_CHANNELS,
    "classes": 1,
    "activation": None,  # Output logits for BCEWithLogitsLoss
    "output_stride": 8,  # Dilated encoder setting
    "decoder_use_batchnorm": True,
    "decoder_channels": [256, 128, 64, 32, 16],
    "decoder_attention_type": "scse",  # Spatial and Channel Squeeze & Excitation
}


# ==========================================
# Utility Functions
# ==========================================
def get_device():
    """Returns the torch device."""
    return torch.device(DEVICE)


def get_working_dir():
    """Returns the working directory path."""
    return WORKING_DIR
