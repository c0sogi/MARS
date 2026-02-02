import os
import torch


class Config:
    # System
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Execution Control
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLES = 100  # Number of samples to use in debug mode

    # Directories
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory & Caching
    # Using idea_13 as specified for this iteration
    WORKING_DIR = "./working/idea_13"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache File Paths (npy format)
    CACHE_TRAIN_IMGS = os.path.join(WORKING_DIR, "cache_train_imgs.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "cache_train_labels.npy")
    CACHE_TRAIN_FILESIZES = os.path.join(WORKING_DIR, "cache_train_filesizes.npy")

    CACHE_VAL_IMGS = os.path.join(WORKING_DIR, "cache_val_imgs.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "cache_val_labels.npy")
    CACHE_VAL_FILESIZES = os.path.join(WORKING_DIR, "cache_val_filesizes.npy")

    CACHE_TEST_IMGS = os.path.join(WORKING_DIR, "cache_test_imgs.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "cache_test_ids.npy")
    CACHE_TEST_FILESIZES = os.path.join(WORKING_DIR, "cache_test_filesizes.npy")

    # Output Paths
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Model Artifacts
    MODEL_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

    # Data Parameters
    IMG_SIZE = 32
    NUM_CLASSES = 1
    # Image Normalization (ImageNet defaults or computed from dataset)
    # Using computed stats from Data Analysis: Mean ~[0.5, 0.45, 0.47], Std ~[0.15, 0.14, 0.15]
    # For simplicity and standard practice, we often use 0.5/0.5 or ImageNet.
    # Let's use simple 0.5 normalization for 32x32 images to keep it robust.
    NORM_MEAN = [0.5, 0.5, 0.5]
    NORM_STD = [0.5, 0.5, 0.5]

    # Training Hyperparameters
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Mixup
    MIXUP_ALPHA = 0.2

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 25
    SWA_LR = 1e-4

    # Stacking / Meta-Learner
    META_LR_C = 1.0  # Inverse regularization strength for Logistic Regression
