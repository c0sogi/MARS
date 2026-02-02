import os
import torch


class Config:
    # ==========================================
    # DIRECTORY SETUP
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for Idea 4 (Full-Scale ResNet-50 + Dual Pooling)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_4")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # FILE PATHS
    # ==========================================
    # Raw Data
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Outputs
    SUBMISSION_PATH = "submission.csv"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model_idea4.pth")

    # ==========================================
    # CACHED ARTIFACTS (DECOUPLED LEARNING)
    # ==========================================
    # These files store the pre-computed embeddings to fit in RAM
    CACHE_TRAIN_FEATURES = os.path.join(CACHE_DIR, "train_features.npy")
    CACHE_TRAIN_LABELS = os.path.join(CACHE_DIR, "train_labels.npy")

    CACHE_VAL_FEATURES = os.path.join(CACHE_DIR, "val_features.npy")
    CACHE_VAL_LABELS = os.path.join(CACHE_DIR, "val_labels.npy")

    CACHE_TEST_FEATURES = os.path.join(CACHE_DIR, "test_features.npy")
    CACHE_TEST_IDS = os.path.join(CACHE_DIR, "test_ids.npy")

    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    BACKBONE_NAME = "resnet50"
    BACKBONE_DIM = 2048

    # Dual-Statistic Pooling: Concatenation of Mean and Max vectors
    # Input Dim = 2048 (Mean) + 2048 (Max) = 4096
    INPUT_DIM = 4096
    NUM_CLASSES = 5270

    # MLP Regularization
    DROPOUT_RATE = 0.25

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42

    # Large batch size since we are training a lightweight MLP on pre-computed features
    BATCH_SIZE = 2048

    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # For AdamW

    # Early Stopping
    PATIENCE = 5

    # ==========================================
    # HARDWARE
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # ==========================================
    # DEBUGGING / DEVELOPMENT
    # ==========================================
    # If True, the data processing pipeline will only process a small subset
    DEBUG = False
    DEBUG_SIZE = 50000
