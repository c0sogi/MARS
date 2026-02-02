import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_FOLDS = 5
    NUM_WORKERS = 4  # Utilizing available vCPUs

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for Idea 13
    WORKING_DIR = "./working/idea_13"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    OOF_DIR = os.path.join(WORKING_DIR, "oof")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data & Augmentation
    # -------------------------------------------------------------------------
    IMG_SIZE = 224
    BATCH_SIZE = (
        32  # Conservative batch size for A100 to handle all architectures safely
    )

    # Augmentation Hyperparameters
    AUG_MIN_SCALE = 0.8  # Minimum scale for RandomResizedCrop
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    PROB_AUG = 0.5  # Probability base for applying mixup/cutmix logic

    # -------------------------------------------------------------------------
    # Model & Training
    # -------------------------------------------------------------------------
    # Tri-Modal Heterogeneous Ensemble
    # Using specific timm registry names for pre-trained weights
    MODEL_ARCHS = [
        "convnext_small.fb_in22k",
        "swin_small_patch4_window7_224.ms_in22k",
        "tf_efficientnetv2_s.in21k",
    ]

    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Intra-Fold Model Soup: Average weights from these specific epochs
    SOUP_EPOCHS = [18, 19, 20]

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed: int = 42):
        """
        Sets the seed for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
