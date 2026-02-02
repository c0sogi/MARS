import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories():
    """
    Creates the working and submission directories if they do not exist.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


class Config:
    # --- Directory Paths ---
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Global Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model Specifications ---
    # Implements the Decoupled Multi-Resolution Hybrid Ensemble strategy.
    # CNNs (ResNet, ConvNeXt) use 256x256 to maximize spatial detail.
    # Swin Transformer uses 224x224 to align with fixed window attention.
    MODEL_SPECS = {
        "resnet50": {
            "model_name": "resnet50.a1_in1k",
            "img_size": 256,
            "pretrained": True,
        },
        "convnext_small": {
            "model_name": "convnext_small.fb_in1k",
            "img_size": 256,
            "pretrained": True,
        },
        "swin_tiny": {
            "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
            "img_size": 224,
            "pretrained": True,
        },
    }
