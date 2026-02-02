import os
import torch


class Config:
    """
    Configuration class for the Dog Breed Classification pipeline.
    Acts as a central source of truth for all hyperparameters and constants.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42

    # Debugging flags to control dataset size for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use if DEBUG is True

    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directory for this specific experimental iteration (Idea 5)
    WORKING_DIR = "./working/idea_5"

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 224
    NUM_CLASSES = 120
    N_FOLDS = 5  # 5-Fold Stratified Cross-Validation

    # ==========================================
    # Model Configuration
    # ==========================================
    # ConvNeXt-Small architecture (pretrained on ImageNet-22k)
    MODEL_NAME = "convnext_small_in22k"

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 30  # T_max=30 for Cosine Annealing
    LR = 1e-4  # Initial learning rate
    WEIGHT_DECAY = 0.01

    # Two-phase training parameters
    FREEZE_EPOCHS = 1  # Epochs to train only the head

    # ==========================================
    # Compute Configuration
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
