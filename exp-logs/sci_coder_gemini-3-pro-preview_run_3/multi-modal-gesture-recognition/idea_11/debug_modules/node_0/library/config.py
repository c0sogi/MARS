import os
import torch


class Config:
    """
    Central configuration for the Boundary-Aware Attentive Kinematic Network (BA-AKN).
    Contains hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    # Windowing
    WINDOW_SIZE = 64
    STRIDE = 32  # 50% overlap for sliding window

    # Labels
    NUM_CLASSES = 21  # 0: Background, 1-20: Gesture Categories
    BACKGROUND_LABEL = 0

    # Feature Dimensions
    # The model expects an Early Fusion vector.
    # We define a standard input dimension that the feature extractor should project to.
    INPUT_DIM = 256

    # ==========================================
    # Model Configuration
    # ==========================================
    # Backbone (Bi-GRU)
    HIDDEN_DIM = 128

    # Refinement Stages (MS-TCN with Attention)
    NUM_STAGES = 3
    NUM_LAYERS = 10
    NUM_F_MAPS = 64
    DROPOUT = 0.3

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Adam optimizer
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50

    # Debugging / Development
    DEBUG = False
    DEBUG_SUBSET_SIZE = 20  # Number of samples to use when DEBUG is True

    # ==========================================
    # Loss Configuration
    # ==========================================
    # Coefficients for Multi-Task Loss
    LAMBDA_BND = 0.5  # Weight for Boundary Detection Loss
    LAMBDA_SMOOTH = 0.15  # Weight for Adaptive Smoothing Loss

    # Class Weights for CrossEntropy
    # Background (Index 0) is weighted 0.2, Gestures (1-20) are weighted 1.0
    CLASS_WEIGHTS = torch.tensor([0.2] + [1.0] * 20, dtype=torch.float32)

    # ==========================================
    # Hardware Configuration
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # For DataLoader
