import os
import torch


class Config:
    """
    Configuration for the Stochastic-Depth Weight-Inflated Volumetric (SD-WIV) Network.
    """

    # ==========================================
    # Reproducibility & System
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for this specific idea/strategy
    WORKING_DIR = "./working/idea_29"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Caching Paths (for deterministic loading)
    # ==========================================
    # We use .npy for large array storage as per requirements
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "cache_train_images.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "cache_train_labels.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "cache_train_ids.npy")

    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "cache_val_images.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "cache_val_labels.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "cache_val_ids.npy")

    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "cache_test_images.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "cache_test_ids.npy")

    # ==========================================
    # Data Preprocessing
    # ==========================================
    IMG_SIZE = 224

    # SD-WIV Strategy: 3 Modalities x 3 Depths = 9 Channels
    SELECTED_MODALITIES = ["FLAIR", "T1wCE", "T2w"]

    # Relative depths within the Brain ROI (Scale-Invariant)
    # 0.4 (Peripheral), 0.5 (Center), 0.6 (Peripheral)
    RELATIVE_DEPTHS = [0.4, 0.5, 0.6]

    NUM_CHANNELS = len(SELECTED_MODALITIES) * len(RELATIVE_DEPTHS)  # 9

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1

    # Weight Inflation Factors (Gaussian Prior)
    # Center slices (Channels 3,4,5) get 50% of original weight energy
    WEIGHT_INFLATION_CENTER = 0.5
    # Peripheral slices (Channels 0,1,2 and 6,7,8) get 25% of original weight energy
    WEIGHT_INFLATION_PERIPHERY = 0.25

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4

    # Regularization
    WEIGHT_DECAY = 1e-2
    CLASSIFIER_DROPOUT = 0.3

    # Structured Depth Dropout (Structural Innovation)
    # Probability to zero out Center (Depth 0.5) or Periphery (Depth 0.4/0.6)
    DEPTH_DROPOUT_PROB = 0.2

    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20
