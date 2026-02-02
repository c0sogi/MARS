import os
import torch


class Config:
    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Leveraging available vCPUs

    # ==========================
    # Directory Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Cache directory for processed numpy arrays (Idea 62 specific)
    CACHE_DIR = "./working/idea_62"

    # Directory for saving model checkpoints
    CHECKPOINT_DIR = "./working/checkpoints"

    # Output submission file path
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================
    # Data Configuration
    # ==========================
    # Native resolution for EfficientNet-B0
    IMAGE_SIZE = 224

    # Number of slabs per view (mapped to RGB channels)
    NUM_SLABS = 3

    # ImageNet Normalization Statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE_NAME = "efficientnet_b0"

    # Dimensionality settings for NDS-SLN
    VISUAL_DIM = 1280  # Output dim of EfficientNet-B0 GAP
    LATENT_DIM = 128  # Shared latent dim for tabular data
    BOTTLENECK_DIM = 64  # Compressed dim for decoupled context streams

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 8

    # Metric clipping thresholds
    MAX_ERROR_CLIP = 1000
    MIN_CONFIDENCE_CLIP = 70
