import os
import torch


class Config:
    """
    Configuration class for Catheter and Line Position Detection.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # --- General Environment ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # --- Debugging ---
    # If True, runs on a small subset of data to verify pipeline functionality
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths (already generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    # Storing artifacts in idea_3 directory as per strategy
    OUTPUT_DIR = "./working/idea_3"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Configuration ---
    # Resolution set to 640 based on optimization lessons
    IMAGE_SIZE = 640

    # --- Model Architecture ---
    # EfficientNetV2-Small backbone
    MODEL_NAME = "tf_efficientnetv2_s"
    PRETRAINED = True

    # Head Configuration
    USE_GEM_POOLING = True  # Generalized Mean Pooling
    GEM_P = 3.0  # Initial power for GeM
    GEM_LEARNABLE = True  # Whether p is learnable
    DROP_RATE = 0.2  # Dropout for linear head
    DROP_PATH_RATE = 0.2  # Stochastic depth rate

    # --- Training Hyperparameters ---
    BATCH_SIZE = 8  # Reduced to fit GPU memory constraints (Cite debug_lesson_1)
    EPOCHS = 10

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3  # Scaled for larger batch size
    WEIGHT_DECAY = 1e-2

    # Scheduler (OneCycleLR)
    PCT_START = 0.1  # Warmup percentage
    DIV_FACTOR = 25.0  # Initial_LR = Max_LR / Div_Factor
    FINAL_DIV_FACTOR = 100.0  # Final_LR = Initial_LR / Final_Div_Factor

    # Early Stopping
    PATIENCE = 4

    # EMA
    USE_EMA = True
    EMA_DECAY = 0.9999

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
    NUM_CLASSES = len(TARGET_COLS)
