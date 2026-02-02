import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Data directories
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output and Cache directories
    # Specific cache for the Tri-Slab idea (Idea 3)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output path
    SUBMISSION_FILE = "./submission/submission.csv"

    # ==========================================
    # Data Configuration
    # ==========================================
    # Image preprocessing
    IMG_SIZE = 240
    SLAB_COUNT = 3  # Number of depth slabs (Top, Middle, Bottom)
    IN_CHANNELS = 3  # RGB channels corresponding to slabs

    # Tabular features to utilize
    # The Dataset class will map specific CSV columns to these semantic features
    TABULAR_FEATURES = ["Age", "Sex", "SmokingStatus", "Percent"]

    # Normalization constants (approximate from EDA)
    AGE_MEAN = 67.0
    AGE_STD = 7.0
    PERCENT_MEAN = 77.0
    PERCENT_STD = 20.0

    # ==========================================
    # Model Configuration
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    # Outputs: alpha (slope), sigma_base, sigma_growth
    NUM_OUTPUTS = 3

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (Cosine Annealing)
    T_MAX = 10
    ETA_MIN = 1e-5

    # Early Stopping
    PATIENCE = 10

    # Hardware
    NUM_WORKERS = 4

    # Debugging
    # Set to True to run on a small subset of data for testing pipeline
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    # ==========================================
    # Metric Configuration
    # ==========================================
    # Laplace Log Likelihood constants
    Q_CLIP = 70.0
    MAX_ERR = 1000.0
