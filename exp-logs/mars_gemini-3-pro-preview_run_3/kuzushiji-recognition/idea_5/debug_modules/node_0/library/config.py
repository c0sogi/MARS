import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Specific to Idea 5)
    WORK_DIR = "./working/idea_5"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    MODEL_DIR = os.path.join(WORK_DIR, "models")
    OUTPUT_DIR = os.path.join(WORK_DIR, "output")
    LOG_DIR = os.path.join(WORK_DIR, "logs")

    # Final Submission Path
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # 2. Global Constants & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================================
    # 3. Data Parameters
    # ==========================================
    # 3848 Kuzushiji characters + 1 Background class
    NUM_CHAR_CLASSES = 3848
    BACKGROUND_CLASS_ID = 3848
    NUM_TOTAL_CLASSES = 3849

    # ==========================================
    # 4. Detector Model (Stage 1)
    # ==========================================
    # Global Context Scale-Preserved Detector
    DETECTOR_ARCH = "resnet34"
    DETECTOR_INPUT_SIZE = 1024  # Square input (longest side resized to 1024, padded)
    DETECTOR_BATCH_SIZE = 16  # Fits in A100 40GB with ResNet34 @ 1024x1024
    DETECTOR_LR = 1e-4
    DETECTOR_EPOCHS = 30

    # ==========================================
    # 5. Classifier Model (Stage 2)
    # ==========================================
    # Verification Classifier with Background Suppression
    CLASSIFIER_ARCH = "resnet50"
    CLASSIFIER_INPUT_SIZE = 128  # High-fidelity crops
    CLASSIFIER_BATCH_SIZE = 64
    CLASSIFIER_LR = 1e-4
    CLASSIFIER_EPOCHS = 15

    # ==========================================
    # 6. Inference & Post-processing
    # ==========================================
    CONF_THRESHOLD = 0.2  # Threshold for detector heatmap
    NMS_IOU_THRESHOLD = 0.3  # IOU threshold for Non-Maximum Suppression
    MAX_DETECTIONS = 1200  # Task constraint: max predictions per page

    # ==========================================
    # 7. Debugging / Development
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of images to use in debug mode

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        directories = [
            cls.WORK_DIR,
            cls.CACHE_DIR,
            cls.MODEL_DIR,
            cls.OUTPUT_DIR,
            cls.LOG_DIR,
            os.path.dirname(cls.SUBMISSION_PATH),
        ]

        for d in directories:
            if d:  # Check if path is not empty
                os.makedirs(d, exist_ok=True)

        # Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Execute setup upon module import
Config.setup()
