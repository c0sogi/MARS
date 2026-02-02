import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==============================
    # Path Configuration
    # ==============================
    ROOT_DIR = "."
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")

    # Metadata CSVs
    TRAIN_META = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_meta.csv")

    # Working Directory (Idea 3 specific)
    WORKING_DIR = os.path.join(ROOT_DIR, "working", "idea_3")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Processed Data Cache Files (Parquet)
    PROCESSED_TRAIN_PKL = os.path.join(WORKING_DIR, "train_processed.parquet")
    PROCESSED_VAL_PKL = os.path.join(WORKING_DIR, "val_processed.parquet")
    PROCESSED_TEST_PKL = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==============================
    # Model Hyperparameters
    # ==============================
    SEED = 42
    BACKBONE_NAME = "resnext101_32x8d"
    # 14 Findings + 1 Background (Class 0 is background in Torchvision)
    NUM_CLASSES = 15

    # ==============================
    # Training Settings
    # ==============================
    # Batch size of 8 is conservative for ResNeXt-101 on 40GB GPU with 1024px images.
    # Can increase to 16 if memory allows.
    BATCH_SIZE = 8
    EPOCHS = 12
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9
    NUM_WORKERS = 4

    # ==============================
    # Data & Image Settings
    # ==============================
    IMG_SIZE = 1024

    # ==============================
    # Inference Settings
    # ==============================
    # Confidence threshold to consider a detection valid before NMS
    CONFIDENCE_THRESHOLD = 0.05
    # IoU threshold for evaluation metric (PASCAL VOC 2010 at IoU > 0.4)
    IOU_THRESHOLD = 0.4

    # ==============================
    # Class Mapping
    # ==============================
    # Maps Dataset Class ID (0-14) to Class Name
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
