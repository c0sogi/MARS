import os
import torch


class Config:
    """
    Configuration for the Two-Stage Stacking Ensemble (Idea 7).
    """

    # --- General Configuration ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    META_LEARNER_PATH = os.path.join(WORKING_DIR, "meta_learner.joblib")

    # Caching Paths (Parquet format)
    CACHED_TRAIN_METADATA = os.path.join(WORKING_DIR, "cached_train_metadata.parquet")
    CACHED_TEST_METADATA = os.path.join(WORKING_DIR, "cached_test_metadata.parquet")
    STACKED_OOF_DATA = os.path.join(WORKING_DIR, "stacked_oof_data.parquet")
    FOLDS_PATH = os.path.join(WORKING_DIR, "folds.parquet")

    # --- Data Processing ---
    # The task requires predicting the center 32x32px.
    # We crop 64x64px to provide 16px context border as per strategy.
    IMAGE_SIZE = 64
    CENTER_CROP_SIZE = 64

    # --- Model Architecture ---
    # Heterogeneous ensemble components
    # 1. ConvNeXt-Tiny: Strong hierarchical features
    # 2. EfficientNetV2-Small: Complementary texture features
    MODEL_BACKBONES = ["convnext_tiny", "tf_efficientnetv2_s"]
    PRETRAINED = True
    NUM_CLASSES = 1

    # --- Training Hyperparameters ---
    N_FOLDS = 5
    NUM_EPOCHS = 30
    # A100 40GB allows large batch sizes for 64x64 images
    BATCH_SIZE = 512

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    EARLY_STOPPING_PATIENCE = 6  # Stop if no improvement for 6 epochs

    # --- Inference & TTA ---
    # Test Time Augmentation settings
    USE_TTA = True

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available; 4 workers per dataloader is usually optimal to avoid overhead
    NUM_WORKERS = 4
