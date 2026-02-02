import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Thoracic Lung Disease Detection Task.
    Handles paths, hyperparameters, and constants for the Faster R-CNN pipeline.
    """

    # --- General ---
    PROJECT_NAME = "idea_2"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available vCPUs for data loading
    NUM_WORKERS = 12

    # --- Directories ---
    INPUT_DIR = "./input"
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Working directory for this specific idea/run
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Cache directory for preprocessed images (PNG/NPY) to decouple ingestion
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model checkpoints and logs
    MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "models")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Preprocessing ---
    IMG_SIZE = 512  # Input resolution for the model

    # Normalization constants (ImageNet defaults)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # --- Model Configuration ---
    # 14 Findings + 1 Background = 15 Classes
    # Note: Dataset class IDs 0-13 map to Model Labels 1-14.
    # Dataset class ID 14 ("No finding") is treated as Background (Label 0).
    NUM_CLASSES = 15

    # Class ID mapping for reference and submission formatting
    CLASS_ID_TO_NAME = {
        0: "Aortic enlargement",
        1: "Atelectasis",
        2: "Calcification",
        3: "Cardiomegaly",
        4: "Consolidation",
        5: "ILD",
        6: "Infiltration",
        7: "Lung Opacity",
        8: "Nodule/Mass",
        9: "Other lesion",
        10: "Pleural effusion",
        11: "Pleural thickening",
        12: "Pneumothorax",
        13: "Pulmonary fibrosis",
        14: "No finding",
    }

    # --- Training Hyperparameters ---
    BATCH_SIZE = 16
    EPOCHS = 12
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    # Learning Rate Scheduler
    WARMUP_EPOCHS = 1

    # Early Stopping
    PATIENCE = 3

    # --- Inference / Post-processing ---
    CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to propose a box
    IOU_THRESHOLD = 0.4  # NMS threshold

    @classmethod
    def setup(cls):
        """
        Create necessary directories for the project.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def seed_everything(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
