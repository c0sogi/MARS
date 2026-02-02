import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for the specific idea
    WORKING_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Gesture Vocabulary
    # 0 is reserved for 'background' / 'silence'
    # 1-20 are the active gestures
    NUM_CLASSES = 21

    # Kinematic Features
    NUM_JOINTS = 20
    # 3 (Pos) + 3 (Vel) + 3 (Acc)
    SKELETON_CHANNELS_PER_JOINT = 9
    SKELETON_INPUT_DIM = NUM_JOINTS * SKELETON_CHANNELS_PER_JOINT  # 180

    # Audio Features
    N_MFCC = 13
    # MFCC + Delta + Delta-Delta
    AUDIO_INPUT_DIM = N_MFCC * 3  # 39

    # Total Input Dimension for Early Fusion
    INPUT_DIM = SKELETON_INPUT_DIM + AUDIO_INPUT_DIM  # 219

    # Windowing
    WINDOW_SIZE = 64
    STRIDE = 32  # 50% overlap for sliding window generation

    # Boundary Label Generation
    # Radius in frames around a transition to label as boundary (1)
    BOUNDARY_RADIUS = 2

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    HIDDEN_DIM = 64
    NUM_STAGES = 3
    DROPOUT = 0.5
    KERNEL_SIZE = 3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 50
    PATIENCE = 10  # For Early Stopping

    # Loss Component Weights
    LAMBDA_CLS = 1.0  # Classification Loss
    LAMBDA_BND = 0.5  # Boundary Prediction Loss
    LAMBDA_SMOOTH = 0.15  # Boundary-Adaptive Smoothing Loss

    # Class Weights
    # Weight for the background class (0) to handle dominance
    BG_WEIGHT = 0.2

    @staticmethod
    def set_seed(seed=None):
        """Sets the random seed for reproducibility."""
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def setup_directories():
        """Ensures all working directories exist."""
        for d in [
            Config.WORKING_DIR,
            Config.CACHE_DIR,
            Config.OUTPUT_DIR,
            Config.SUBMISSION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)
