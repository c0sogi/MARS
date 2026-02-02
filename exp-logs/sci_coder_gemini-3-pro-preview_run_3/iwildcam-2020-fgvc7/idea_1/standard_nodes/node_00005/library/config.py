import os
import torch


class Config:
    """
    Configuration class for the Camera Trap Animal Detection project.
    Defines paths, hyperparameters, and constants used across the pipeline.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Directory for caching intermediate data (e.g., bounding boxes, processed arrays)
    # Using 'idea_1' as a namespace for this specific approach
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Specific File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    MEGADETECTOR_PATH = os.path.join(
        INPUT_DIR, "iwildcam2020_megadetector_results.json"
    )
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "mobilenetv3_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    # Image resolution for MobileNetV3 input (Standard is 224x224)
    IMG_SIZE = (224, 224)

    # Batch size for training and inference
    BATCH_SIZE = 64

    # Number of data loading workers
    NUM_WORKERS = 4

    # MegaDetector Confidence Threshold
    # Images with max_detection_conf < CONF_THRESHOLD are classified as Empty (Class 0)
    # and excluded from the neural network training/inference loop.
    CONF_THRESHOLD = 0.2

    # Number of classes (0 to 675 based on sample submission)
    NUM_CLASSES = 676

    # Class ID representing 'Empty'
    EMPTY_CLASS_ID = 0

    # ==========================================
    # Training Parameters
    # ==========================================
    MODEL_NAME = "mobilenet_v3_small"
    PRETRAINED = True

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Total training epochs
    EPOCHS = 12

    # Early stopping patience (epochs without improvement)
    EARLY_STOPPING_PATIENCE = 5

    # Computation Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working, cache, and submission directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
