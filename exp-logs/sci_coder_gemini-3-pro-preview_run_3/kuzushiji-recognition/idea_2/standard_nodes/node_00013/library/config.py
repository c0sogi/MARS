import os
import torch
import random
import numpy as np


class Config:
    """
    Global configuration for the Kuzushiji Recognition pipeline.
    """

    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (for deterministic data processing)
    # These will be used by dataset modules to store processed arrays/mappings
    CACHE_CLASS_MAP = os.path.join(WORKING_DIR, "class_map.npy")
    CACHE_DETECTOR_TRAIN = os.path.join(WORKING_DIR, "detector_train.npy")
    CACHE_DETECTOR_VAL = os.path.join(WORKING_DIR, "detector_val.npy")
    CACHE_CLASSIFIER_TRAIN = os.path.join(WORKING_DIR, "classifier_train.npy")
    CACHE_CLASSIFIER_VAL = os.path.join(WORKING_DIR, "classifier_val.npy")

    # Model Checkpoints
    DETECTOR_MODEL_PATH = os.path.join(WORKING_DIR, "detector_best.pth")
    CLASSIFIER_MODEL_PATH = os.path.join(WORKING_DIR, "classifier_best.pth")

    # ==========================================
    # 2. Data Parameters
    # ==========================================
    # Stage 1: Detector
    DETECTOR_IMG_SIZE = 1024  # Input resolution for the page-level detector
    DETECTOR_OUTPUT_STRIDE = 4  # Downsampling factor of the backbone (e.g., ResNet/FPN)

    # Stage 2: Classifier
    CLASSIFIER_IMG_SIZE = 64  # Input resolution for character crops

    # Normalization (ImageNet defaults)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # Labels
    # Total unique codes in unicode_translation.csv is 4782.
    # Training set has ~3848. We will handle mapping dynamically,
    # but this constant can serve as an upper bound or vocab size.
    NUM_CLASSES = 4782

    # Inference Constraints
    MAX_PREDICTIONS = 1200  # Maximum number of predictions per page
    CONF_THRESHOLD = 0.2  # Minimum heatmap confidence to consider a point

    # ==========================================
    # 3. Training Hyperparameters
    # ==========================================
    SEED = 42

    # Debugging / Quick Prototyping
    DEBUG = False  # Set to True to train on a small subset
    DEBUG_SAMPLE_SIZE = 100

    # Detector Training (Stage 1)
    DETECTOR_BATCH_SIZE = 16  # Fits on A100 with 1024x1024 images
    DETECTOR_LR = 1e-4
    DETECTOR_EPOCHS = 30
    DETECTOR_PATIENCE = 5  # Early stopping patience

    # Classifier Training (Stage 2)
    CLASSIFIER_BATCH_SIZE = 256  # Small crops allow large batches
    CLASSIFIER_LR = 1e-3
    CLASSIFIER_EPOCHS = 20
    CLASSIFIER_PATIENCE = 3

    # ==========================================
    # 4. System Settings
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
