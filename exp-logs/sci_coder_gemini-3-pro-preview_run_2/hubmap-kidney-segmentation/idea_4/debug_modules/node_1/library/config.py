import os
import torch


class Config:
    """
    Configuration class for the HuBMAP Kidney Segmentation pipeline.
    Stores hyperparameters, file paths, and model settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ====================================================
    # Directories
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Output paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ====================================================
    # Data Preprocessing
    # ====================================================
    TILE_SIZE = 1024
    # Representative Undersampling: Keep 100% of positive tiles, sample 20% of negative tiles
    NEGATIVE_SAMPLE_RATE = 0.20

    # ====================================================
    # Model Architecture
    # ====================================================
    MODEL_NAME = "FPN"
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3

    # Two classes for the internal model:
    # 0: Background
    # 1: Glomerulus (Primary Target)
    # 2: Anatomical Structure (Auxiliary Target - handled via multi-head or multi-channel)
    # Strategy: The model will output 2 channels. Channel 0 = Glomerulus, Channel 1 = Cortex.
    NUM_CLASSES = 2

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    EPOCHS = 20
    WARMUP_EPOCHS = 5  # Epochs before saving best model / enabling early stopping

    BATCH_SIZE = 4  # Adjusted for 1024x1024 tiles on A100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Loss Weights
    # L_total = L_glom + lambda * L_cortex
    AUX_LOSS_WEIGHT = 0.5

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 5

    # ====================================================
    # Inference
    # ====================================================
    # Sliding window inference
    INFERENCE_OVERLAP = 0.5

    # Thresholding
    MASK_THRESHOLD = 0.5

    # Post-processing
    MIN_PIXEL_SIZE = 50  # Remove small artifacts

    # ====================================================
    # Hardware
    # ====================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
