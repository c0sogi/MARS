import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # --- General ---
    SEED = 42
    DEBUG = False  # Set to True to limit epochs and data for debugging

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # specific cache directory as requested
    OUTPUT_DIR = os.path.join(WORKING_DIR, "idea_9")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data Paths ---
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    BOUNDING_BOX_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output submission path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Caching ---
    CACHE_DIR = OUTPUT_DIR
    USE_CACHE = True

    # --- Data Processing ---
    IMAGE_SIZE = 384
    IN_CHANNELS = 3  # 2.5D stacking: slices [z-1, z, z+1]
    SEQ_LEN = 96  # Number of slices sampled per study
    NUM_WORKERS = 4  # 12 vCPUs available, 4 is a safe balance

    # --- Model Architecture ---
    BACKBONE = "efficientnet_b4"
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.2
    NUM_CLASSES = 8  # C1-C7 + patient_overall

    # --- Training ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch size is small due to large volumetric data (Batch x Seq x C x H x W)
    # Batch size reduced to prevent OOM (Cite debug_lesson_1)
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8  # Effective batch size = 16

    EPOCHS = 10 if not DEBUG else 2
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000

    # Learning Rate Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 3

    # --- Loss Weights ---
    # Total Loss = L_study + (lambda_slice * L_slice) + (lambda_spatial * L_spatial)
    LAMBDA_SLICE = 1.0
    LAMBDA_SPATIAL = 1.0
    POS_WEIGHT_STUDY = 2.0  # Weight for positive class in study-level loss

    # --- Targets ---
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]


# Initialize seeds on import
seed_everything(Config.SEED)
