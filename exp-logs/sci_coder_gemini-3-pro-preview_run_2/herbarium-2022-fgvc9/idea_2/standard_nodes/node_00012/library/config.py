import os
import torch


class Config:
    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilize available vCPUs

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Input Data Directories and Files
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train_metadata.json")
    TEST_METADATA_JSON = os.path.join(INPUT_DIR, "test_metadata.json")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Generated Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output & Caching
    # Caching directory for deterministic data processing (e.g. taxonomy maps)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "efficientnet_b3"
    # Resize images to 260x260 (native resolution for EfficientNet-B3 as per strategy)
    IMAGE_SIZE = 260
    NUM_CLASSES = 15501
    DROPOUT_RATE = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # High throughput settings for A100
    BATCH_SIZE = 32
    NUM_EPOCHS = 35
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early Stopping

    # Hierarchical Multi-Task Loss Weights
    LOSS_WEIGHTS = {"species": 1.0, "genus": 0.5, "family": 0.5}

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000

    @classmethod
    def create_directories(cls):
        """Creates necessary working and cache directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
