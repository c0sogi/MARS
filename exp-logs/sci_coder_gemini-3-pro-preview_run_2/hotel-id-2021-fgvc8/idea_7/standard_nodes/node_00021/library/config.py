import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for Hotel Identification (Idea 7).
    Encapsulates paths, model hyperparameters, training settings, and inference parameters.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for Graph-Based Regularization)
    GALLERY_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "gallery_embeddings.npy")
    GALLERY_LABELS_PATH = os.path.join(WORKING_DIR, "gallery_labels.npy")
    QUERY_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "query_embeddings.npy")
    QUERY_NAMES_PATH = os.path.join(WORKING_DIR, "query_names.npy")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 384  # High resolution for fine-grained details
    N_CLASSES = 7770  # Total unique hotels in training set
    BATCH_SIZE = 16  # Adjusted for 16GB GPU with ConvNeXt-Small @ 384x384
    NUM_WORKERS = 8  # Optimal for 12 vCPUs
    PIN_MEMORY = True

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "convnext_small"
    PRETRAINED = True
    EMBEDDING_SIZE = 512
    DROPOUT = 0.0  # Minimal dropout for embeddings

    # Sub-Center ArcFace Head Parameters
    MARGIN = 0.50
    SCALE = 30.0
    K_SUB_CENTERS = 3  # K=3 sub-centers to handle intra-class variance
    LABEL_SMOOTHING = 0.0  # Disabled to ensure sharp angular margins

    # =========================================================================
    # Training Configuration
    # =========================================================================
    SEED = 42
    EPOCHS = 18  # Extended training for margin convergence
    LR = 1e-4  # Backbone learning rate
    HEAD_LR = 1e-3  # Often higher LR for the classification head
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6
    SCHEDULER_TYPE = "CosineAnnealingLR"

    # =========================================================================
    # Inference & Post-Processing (Graph Regularization)
    # =========================================================================
    KNN = 50  # Number of neighbors for DBA/QE
    TOP_K = 5  # Number of predictions per image (MAP@5)
    USE_DBA = True  # Enable Database Augmentation
    USE_QE = True  # Enable Query Expansion

    # =========================================================================
    # Debugging and Runtime
    # =========================================================================
    DEBUG = False  # Set True to run on a small subset
    DEBUG_SAMPLE_SIZE = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Performs initial setup: creates necessary directories and sets random seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """
        Sets the seed for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
