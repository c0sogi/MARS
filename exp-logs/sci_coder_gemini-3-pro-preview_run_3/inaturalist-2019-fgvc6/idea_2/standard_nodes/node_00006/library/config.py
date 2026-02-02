import os
import torch

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------

# Random Seed for Reproducibility
SEED = 42

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Utilizing all available vCPUs
PIN_MEMORY = True

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Dataset Specifics
NUM_CLASSES = 1010
DEBUG_SAMPLE_SIZE = (
    None  # Set to integer (e.g., 2000) for quick debugging, None for full training
)

# Model Architecture
# Using EfficientNetV2-M for better capacity on fine-grained tasks compared to Small variant
MODEL_NAME = "tf_efficientnetv2_m.in21k_ft_in1k"

# -----------------------------------------------------------------------------
# Stage 1: Representation Learning (Low Resolution)
# -----------------------------------------------------------------------------
# Objective: Learn robust features on 224x224 images using standard sampling.
STAGE_1_CONFIG = {
    "stage_name": "stage_1_representation",
    "image_size": 224,
    "batch_size": 64,  # Higher batch size for efficiency at lower resolution
    "epochs": 6,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patience": 2,  # Early stopping patience
    "backbone_frozen": False,
    "sampling_strategy": "instance_balanced",  # Standard random sampling
    "label_smoothing": 0.1,  # Helps with generalization
    "rand_augment": True,
    "rand_augment_num_ops": 2,
    "rand_augment_magnitude": 9,
    "checkpoint_name": "stage_1_best.pth",
    "load_from": None,
}

# -----------------------------------------------------------------------------
# Stage 2: High-Resolution Adaptation
# -----------------------------------------------------------------------------
# Objective: Fine-tune the model on 384x384 images to resolve fine details.
STAGE_2_CONFIG = {
    "stage_name": "stage_2_high_res",
    "image_size": 384,
    "batch_size": 24,  # Reduced batch size for high resolution (fits A100 40GB)
    "epochs": 4,
    "learning_rate": 1e-4,  # Reduced LR for fine-tuning
    "weight_decay": 1e-4,
    "patience": 2,
    "backbone_frozen": False,
    "sampling_strategy": "instance_balanced",
    "label_smoothing": 0.0,  # Switch to standard Cross Entropy
    "rand_augment": True,
    "rand_augment_num_ops": 2,
    "rand_augment_magnitude": 9,
    "checkpoint_name": "stage_2_best.pth",
    "load_from": "stage_1_best.pth",
}

# -----------------------------------------------------------------------------
# Stage 3: Decoupled Classifier Alignment (Long-Tail Handling)
# -----------------------------------------------------------------------------
# Objective: Calibrate classifier head for rare classes while preserving features.
STAGE_3_CONFIG = {
    "stage_name": "stage_3_classifier_alignment",
    "image_size": 384,
    "batch_size": 32,
    "epochs": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 2,
    "backbone_frozen": True,  # Freeze backbone to preserve learned features
    "sampling_strategy": "class_balanced",  # Oversample rare classes
    "label_smoothing": 0.0,
    "rand_augment": True,
    "rand_augment_num_ops": 2,
    "rand_augment_magnitude": 9,
    "checkpoint_name": "stage_3_best.pth",
    "load_from": "stage_2_best.pth",
}
