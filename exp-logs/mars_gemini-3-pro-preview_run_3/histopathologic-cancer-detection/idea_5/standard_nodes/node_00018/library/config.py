import os
import torch


class Config:
    # --- Project & Paths ---
    PROJECT_NAME = "idea_5"

    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Create working and submission directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata file paths
    # Note: For 5-fold CV on 100% data, the training pipeline will likely need to
    # combine train.csv and val.csv, or we use train.csv as the primary source.
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Processing ---
    # The task requires predicting the center 32x32px region.
    # We use a 64x64px crop to provide context (16px buffer around the 32px ROI).
    IMG_SIZE = 64
    CENTER_CROP_SIZE = 64

    # --- Model Architecture ---
    # Heterogeneous Ensemble: ConvNeXt (Transformer-style CNN) + DenseNet (Concatenative CNN)
    MODELS = ["convnext_tiny", "densenet121"]

    # --- Training Hyperparameters ---
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 30

    # A100 40GB allows for a large batch size with 64x64 images
    BATCH_SIZE = 512

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # --- Hardware & Compute ---
    NUM_WORKERS = 12  # Matches available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Inference ---
    # Test Time Augmentation steps
    # 4 views: Original, Horizontal Flip, Vertical Flip, Combined Flip
    TTA_STEPS = 4
