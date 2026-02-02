import os
import torch


class Config:
    """
    Configuration class for the Catheter and Line Detection task.
    """

    # --- General ---
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Create output directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- File Paths ---
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data ---
    # Cite solution_lesson_node_00004: Increase resolution to 640x640 for better thin-feature detection
    IMAGE_SIZE = 640
    NUM_WORKERS = 12  # Utilizing the 12 vCPUs available

    # --- Model ---
    # Cite solution_lesson_node_00004: Upgrade to ResNet34
    MODEL_NAME = "resnet34"
    PRETRAINED = True
    NUM_CLASSES = 11

    # --- Training ---
    # A100 40GB can handle larger batches for ResNet18 @ 512x512
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    MAX_LR = 1e-3  # For OneCycleLR scheduler
    WEIGHT_DECAY = 1e-2

    # --- Compute ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Target Labels ---
    TARGET_COLS = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]
