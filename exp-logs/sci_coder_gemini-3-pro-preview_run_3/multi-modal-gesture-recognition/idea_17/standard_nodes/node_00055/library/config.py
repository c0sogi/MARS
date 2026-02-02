import os
import torch


class Config:
    # ==========================================
    # Project & Directory Settings
    # ==========================================
    PROJECT_NAME = "idea_18"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Reproducibility
    SEED = 42

    # Sampling
    WINDOW_SIZE = 64
    STRIDE = 32  # Moderate stride to prevent redundancy (Lesson 00053)
    TEST_STRIDE = 32  # For sliding window inference (50% overlap)

    # Skeleton Structure
    NUM_JOINTS = 20
    # Features: Position (3) + Velocity (3) + Acceleration (3)
    # We exclude bone vectors to minimize dimensionality (Lesson 00048)
    SKELETON_FEATURE_DIM = NUM_JOINTS * 3 * 3  # 180

    # Audio Structure
    NUM_MFCC = 13

    # Total Input Dimension for Early Fusion
    INPUT_DIM = SKELETON_FEATURE_DIM + NUM_MFCC  # 193

    # Labels
    # 20 Gesture Classes + 1 Background Class (0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # Model Architecture
    # ==========================================
    # Encoder
    HIDDEN_DIM = 128
    GRU_LAYERS = 2
    DROPOUT = 0.3

    # Refinement (Independent TCN Stages)
    TCN_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    # Reduced max dilation to control capacity with independent stages (Cite solution_lesson_node_00053)
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    NUM_REFINEMENT_STAGES = 2  # Number of independent stages

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    PATIENCE = 10  # Early stopping patience

    # Loss Weights
    # Weight for background class (0.2) vs others (1.0) (Lesson 00010)
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = 0.2

    # Smoothing Loss
    MSE_SMOOTHING_WEIGHT = 0.15

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed():
        """Sets fixed random seeds for reproducibility."""
        import random
        import numpy as np

        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
