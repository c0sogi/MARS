import os
import torch

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea as per requirements
WORKING_DIR = "./working/idea_14"
os.makedirs(WORKING_DIR, exist_ok=True)

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Settings
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# We have 12 vCPUs, so 4-8 workers is usually safe and efficient
NUM_WORKERS = 4

# ==========================================
# Data Configuration
# ==========================================
# Teacher models (ResNet, ConvNeXt) benefit from larger resolution (Pipeline A)
IMG_SIZE_TEACHER = 256
# Student model (MaxViT) uses native resolution (Pipeline B)
IMG_SIZE_STUDENT = 224

BATCH_SIZE = 32

# Standard ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# Model Architecture Configuration
# ==========================================
# Using timm model names
MODEL_RESNET = "resnet50.a1_in1k"
MODEL_CONVNEXT = "convnext_small.fb_in1k"
MODEL_MAXVIT = "maxvit_tiny_tf_224.in1k"

# ==========================================
# Training Hyperparameters
# ==========================================
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3  # Early stopping patience

# Distillation Parameters
# Loss = alpha * BCE + (1 - alpha) * KLDiv
DISTILLATION_ALPHA = 0.5
DISTILLATION_TEMP = 1.0

# ==========================================
# Augmentation Hyperparameters
# ==========================================
# RandomResizedCrop scale to ensure subject is not lost
AUG_CROP_SCALE = (0.8, 1.0)
# ColorJitter intensity
AUG_COLOR_JITTER_INTENSITY = 0.2

# ==========================================
# Debugging / Development
# ==========================================
# Set DEBUG to True to limit dataset size for quick pipeline validation
DEBUG = False
DEBUG_SAMPLE_SIZE = 200
