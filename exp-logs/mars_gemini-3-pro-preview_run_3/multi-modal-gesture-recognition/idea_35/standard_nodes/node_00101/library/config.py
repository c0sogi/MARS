import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Global Seeding
    # ==========================================
    SEED = 42

    def set_seed(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = "./working/idea_35"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Parameters
    # ==========================================
    # 20 Gestures + 1 Background (Class 0)
    NUM_CLASSES = 21

    # Input Features
    # 20 joints * 3 coords (pos) + 20*3 (vel) + 20*3 (acc) = 180
    # Plus Audio MFCC (e.g., 13 or 20 coeffs). Let's assume 13 MFCCs + 1 Energy = 14, or similar.
    # The data loader will determine exact input size, but we define skeleton components here.
    JOINTS_COUNT = 20

    # Sliding Window Strategy
    WINDOW_SIZE = 64
    STRIDE = 32

    # ==========================================
    # Model Architecture
    # ==========================================
    # Stage 1: Bi-GRU Encoder
    RNN_HIDDEN_SIZE = 128  # Per direction
    RNN_BIDIRECTIONAL = True
    RNN_LAYERS = 2
    DROPOUT_RNN = 0.3

    # Stage 2 & 3: TCN Refinement
    TCN_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    # Monotonically increasing dilation schedule
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    DROPOUT_TCN = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights
    # Background class (0) gets 0.2 weight, others 1.0
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[0] = 0.2

    # Deep Supervision Weights
    LOSS_WEIGHT_STAGE1 = 1.0
    LOSS_WEIGHT_STAGE2 = 1.0
    LOSS_WEIGHT_STAGE3 = 1.0

    # Log-Space Smoothing Loss
    SMOOTHING_LOSS_WEIGHT = 0.15
    LOG_MSE_THRESHOLD = 1.0

    # ==========================================
    # Post-Processing
    # ==========================================
    # Minimum duration (in frames) to keep a gesture prediction
    MIN_GESTURE_DURATION = 5

    # ==========================================
    # Label Map
    # ==========================================
    # Maps internal class ID (1-20) to string name. 0 is background.
    ID_TO_NAME = {
        1: "vattene",
        2: "vieniqui",
        3: "perfetto",
        4: "furbo",
        5: "cheduepalle",
        6: "chevuoi",
        7: "daccordo",
        8: "seipazzo",
        9: "combinato",
        10: "freganiente",
        11: "ok",
        12: "cosatifarei",
        13: "basta",
        14: "prendere",
        15: "noncenepiu",
        16: "fame",
        17: "tantotempo",
        18: "buonissimo",
        19: "messidaccordo",
        20: "sonostufo",
    }

    @classmethod
    def get_class_weights(cls, device):
        return cls.CLASS_WEIGHTS.to(device)


# Instantiate to run setup logic immediately upon import
config = Config()
config.set_seed()
