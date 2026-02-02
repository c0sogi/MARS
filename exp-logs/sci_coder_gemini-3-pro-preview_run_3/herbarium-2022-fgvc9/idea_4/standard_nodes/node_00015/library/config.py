import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # --------------------------------------------------------------------------
    # Directories and Paths
    # --------------------------------------------------------------------------
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata files
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw metadata for hierarchy extraction (Family/Genus mapping)
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train_metadata.json")

    # Working directory for artifacts (checkpoints, cache)
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 256
    BATCH_SIZE = 32
    NUM_WORKERS = 12  # Utilizing all available vCPUs
    PIN_MEMORY = True

    # --------------------------------------------------------------------------
    # Model Parameters
    # --------------------------------------------------------------------------
    BACKBONE = "tf_efficientnetv2_s"
    PRETRAINED = True
    NUM_CLASSES = 15501  # Species (Primary Task)

    # --------------------------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------------------------
    EPOCHS = 18

    # Optimizer & Scheduler (OneCycleLR)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Regularization
    LABEL_SMOOTHING = 0.1

    # Multi-Task Loss Weights
    # Total Loss = L_species + 0.1 * L_genus + 0.1 * L_family
    LOSS_WEIGHTS = {"species": 1.0, "genus": 0.1, "family": 0.1}

    # --------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------
    USE_TTA = True  # Horizontal Flip Test Time Augmentation

    # --------------------------------------------------------------------------
    # Hardware
    # --------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
