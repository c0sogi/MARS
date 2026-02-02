import os
import torch

# =============================================================================
# Global Constants
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Batch size adjusted for 40GB A100 GPU with Large models
BATCH_SIZE = 32
NUM_CLASSES = 120
# Use available vCPUs for data loading
NUM_WORKERS = 12

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea to ensure cache isolation
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Data Normalization (for Torchvision Stream)
# =============================================================================
# Standard ImageNet Mean and Std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Stream Configurations
# =============================================================================
# Defines the architecture and preprocessing logic for the Dual-Stream Ensemble
STREAMS = {
    "stream_a": {
        "name": "stream_a",
        "library": "torchvision",
        "model_name": "convnext_large",
        "weights": "IMAGENET1K_V1",
        # Multi-View Geometric Transformations
        "views": {
            "global": {"resize": (224, 224), "crop": None, "flip": True},  # Squish
            "standard": {"resize": 232, "crop": 224, "flip": True},  # Standard
            "local": {"resize": 288, "crop": 224, "flip": True},  # Zoom
        },
    },
    "stream_b": {
        "name": "stream_b",
        "library": "timm",
        "model_name": "maxvit_large_tf_224.in1k",
        "pretrained": True,
        # Multi-View Geometric Transformations
        "views": {
            "global": {"resize": (224, 224), "crop": None, "flip": True},
            "standard": {"resize": 232, "crop": 224, "flip": True},
            "local": {"resize": 288, "crop": 224, "flip": True},
        },
    },
}


# =============================================================================
# Path Helper Functions
# =============================================================================
def get_embedding_path(stream_name, split, view):
    """
    Generates the path for caching embeddings.
    Format: ./working/idea_15/{stream}_{split}_{view}_embeddings.npy
    """
    filename = f"{stream_name}_{split}_{view}_embeddings.npy"
    return os.path.join(WORKING_DIR, filename)


def get_ids_path(stream_name, split, view):
    """
    Generates the path for caching IDs corresponding to embeddings.
    """
    filename = f"{stream_name}_{split}_{view}_ids.npy"
    return os.path.join(WORKING_DIR, filename)


def get_labels_path(stream_name, split, view):
    """
    Generates the path for caching labels corresponding to embeddings.
    """
    filename = f"{stream_name}_{split}_{view}_labels.npy"
    return os.path.join(WORKING_DIR, filename)


def get_model_head_path(stream_name):
    """
    Generates the path for saving the trained Logistic Regression head.
    """
    filename = f"{stream_name}_logreg_head.joblib"
    return os.path.join(WORKING_DIR, filename)
