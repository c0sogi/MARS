import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Thoracic Lung Disease Detection Task.
    Defines paths, model hyperparameters, training settings, and inference thresholds.
    """

    # ==============================
    # Path Configuration
    # ==============================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Working directory for caching processed data and saving model checkpoints
    WORKING_DIR = "./working/idea_1"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Model Configuration
    # ==============================
    MODEL_NAME = "retinanet_resnet50_fpn"
    BACKBONE = "resnet50"

    # Number of classes for the model.
    # The dataset has 14 finding classes (0-13) + 1 "No finding" class (14).
    # Torchvision detection models use class 0 for background.
    # We map dataset classes 0-13 to model classes 1-14.
    # Total model classes = 1 (background) + 14 (findings) = 15.
    NUM_CLASSES = 15

    PRETRAINED = True

    # ==============================
    # Data Configuration
    # ==============================
    IMAGE_SIZE = 640
    # Batch size optimized for A100 40GB GPU
    BATCH_SIZE = 16
    NUM_WORKERS = 8

    # ==============================
    # Training Configuration
    # ==============================
    EPOCHS = 15
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.9
    SEED = 42
    EARLY_STOPPING_PATIENCE = 4

    # ==============================
    # Inference Configuration
    # ==============================
    # Minimum confidence score to consider a detection valid
    CONF_THRESHOLD = 0.01
    # IoU threshold for Non-Maximum Suppression (NMS)
    NMS_IOU_THRESHOLD = 0.5

    # ==============================
    # Class Mapping
    # ==============================
    # Mapping from Dataset Class ID to Human-Readable Name
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

    @classmethod
    def setup(cls):
        """
        Prepares the environment by creating necessary directories and setting random seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        cls.seed_everything(cls.SEED)

    @staticmethod
    def seed_everything(seed: int):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

        Args:
            seed (int): The seed value to use.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
