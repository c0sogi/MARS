import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data
    DEBUG_SAMPLES = 100  # Number of samples to use in debug mode

    # --------------------------------------------------------------------------
    # Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (checkpoints, cache, etc.)
    # Using idea_4 to isolate this specific experiment's outputs
    WORKING_DIR = "./working/idea_4"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "convnext_base_best.pth")
    FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    NUM_CLASSES = 120
    RESIZE_SIZE = 256
    CROP_SIZE = 224

    # Compute settings
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # --------------------------------------------------------------------------
    # Model Configuration
    # --------------------------------------------------------------------------
    # Using ConvNeXt-Base pre-trained on ImageNet-1k (Cite solution_lesson_node_00014)
    MODEL_NAME = "convnext_base.fb_in1k"
    HEAD_DROPOUT = 0.5  # Dropout rate for the classification head

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase 1: Head Adaptation (Frozen Backbone)
    PHASE1_EPOCHS = 4
    PHASE1_LR = 1e-3

    # Phase 2: Fine-Tuning (Full Model)
    # Using a longer duration with early stopping
    PHASE2_EPOCHS = 30
    PHASE2_LR_HEAD = 1e-4
    PHASE2_LR_BACKBONE = 1e-6

    # Optimization
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early stopping patience

    # Scheduler (Cosine Annealing)
    # T_max will be set to PHASE2_EPOCHS in the training loop
    ETA_MIN = 1e-7
