import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Cervical Spine Fracture Detection task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # --- General Settings ---
    PROJECT_NAME = "Cervical_Spine_Fracture_Detection"
    IDEA_NAME = "idea_4"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 10  # Number of samples to use in debug mode

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # --- Paths ---
    # Input Directories
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output / Working Directory
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- Data Preprocessing ---
    # 2.5D Stacking: We use 3 channels (z-1, z, z+1)
    IN_CHANS = 3
    # High resolution as per strategy
    IMAGE_SIZE = (384, 384)
    # Sequence length for LSTM
    SEQ_LEN = 64

    # --- Model Architecture ---
    # Backbone: EfficientNet-B4 for high capacity
    BACKBONE = "tf_efficientnet_b4_ns"
    # LSTM Settings
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.1
    # Attention Settings
    ATTENTION_HEADS = 8  # One for each class (C1-C7 + overall)
    EMBED_DIM = 128  # Dimension for positional embeddings

    # --- Targets ---
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    NUM_CLASSES = len(TARGET_COLS)

    # --- Training Hyperparameters ---
    EPOCHS = 10
    # Physical batch size (small due to large image size and B4 backbone)
    BATCH_SIZE = 4
    # Gradient Accumulation to simulate larger batch size (Effective BS = 4 * 4 = 16)
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 10.0

    # Loss Weights
    # Positive class weight > 2.0 to prioritize sensitivity to fractures
    POS_WEIGHT = 2.5

    # Early Stopping
    PATIENCE = 3
    MIN_DELTA = 0.001


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
