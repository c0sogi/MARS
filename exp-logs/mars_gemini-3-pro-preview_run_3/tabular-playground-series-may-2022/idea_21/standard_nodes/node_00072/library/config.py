import os
import torch


class Config:
    """
    Configuration for Regularization-Spectrum Parallel Funnel Ensemble (RSPFE).
    Defines hyperparameters, paths, and architectural settings for the 5-stream ensemble.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run with a subset of data for debugging
    DEBUG_SAMPLES = 50000

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Input Files (Metadata)
    # Using metadata splits as requested
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Directory for processed data
    CACHE_DIR = WORKING_DIR

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Available vCPUs: 12. Using a reasonable number for dataloaders.
    NUM_WORKERS = 4

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    # Feature Definitions
    # Categorical: f_27 (decomposed into 10 chars) + f_29 + f_30
    CAT_FEATURES = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: f_00 to f_26, f_28, and the engineered feature 'unique_character_count'
    CONT_FEATURES = [f"f_{i:02d}" for i in range(27)] + [
        "f_28",
        "unique_character_count",
    ]

    # Embedding Dimensions for all categorical features
    EMBEDDING_DIM = 16

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 30

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3  # Max LR for OneCycleLR
    WEIGHT_DECAY = 1e-5

    # Scheduler (OneCycleLR) settings
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # ==========================================
    # Model Architecture (RSPFE)
    # ==========================================
    # 5 Independent Streams with varying capacity and regularization
    # All streams use ReLU activation and output to a single neuron.
    STREAMS = [
        # Stream 1: Anchor (Standard Funnel, Moderate Dropout)
        {"id": "stream_1_anchor", "layers": [512, 256, 128], "dropout": 0.20},
        # Stream 2: Anchor (Standard Funnel, Moderate Dropout)
        {"id": "stream_2_anchor", "layers": [512, 256, 128], "dropout": 0.20},
        # Stream 3: High Capacity (Wide Funnel, Higher Dropout)
        {"id": "stream_3_high_cap", "layers": [1024, 512, 256], "dropout": 0.25},
        # Stream 4: Conservative (Standard Funnel, High Dropout)
        {"id": "stream_4_conservative", "layers": [512, 256, 128], "dropout": 0.30},
        # Stream 5: Aggressive (Standard Funnel, Low Dropout)
        {"id": "stream_5_aggressive", "layers": [512, 256, 128], "dropout": 0.10},
    ]
