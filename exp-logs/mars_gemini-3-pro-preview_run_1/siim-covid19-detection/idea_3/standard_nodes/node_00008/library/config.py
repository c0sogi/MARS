import os
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create submission dir if it doesn't exist (though usually handled by submission script)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 512
    NUM_CHANNELS = 3  # EfficientNet expects RGB

    # Caching Paths (for data_processing module)
    TRAIN_CACHE_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
    TRAIN_CACHE_MASKS = os.path.join(WORKING_DIR, "train_masks.npy")
    TRAIN_CACHE_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    VAL_CACHE_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
    VAL_CACHE_MASKS = os.path.join(WORKING_DIR, "val_masks.npy")
    VAL_CACHE_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    TEST_CACHE_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
    TEST_CACHE_DIMS = os.path.join(WORKING_DIR, "test_dims.parquet")

    # =========================================================================
    # Model Configuration
    # =========================================================================
    ENCODER_NAME = "efficientnet_b3"
    ENCODER_PRETRAINED = True
    NUM_STUDY_CLASSES = 4

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4

    # Loss Weights
    SEG_LOSS_WEIGHT = 1.0
    CLS_LOSS_WEIGHT = 1.0

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Labels & Mappings
    # =========================================================================
    STUDY_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    LABEL2ID = {label: i for i, label in enumerate(STUDY_LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(STUDY_LABELS)}

    # =========================================================================
    # Debugging
    # =========================================================================
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100
