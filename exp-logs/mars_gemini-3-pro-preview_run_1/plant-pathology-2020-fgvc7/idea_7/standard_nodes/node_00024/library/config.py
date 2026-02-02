import os
import torch


class Config:
    # ==========================================
    # Directories & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files (using generated metadata)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Configuration
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = 4

    # ==========================================
    # Compute Configuration
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True

    # ==========================================
    # Training Configuration (Progressive Resizing)
    # ==========================================

    # Phase 1: Coarse Training
    # Focus on structural features with smaller resolution and higher LR
    IMG_SIZE_PHASE_1 = 256
    BATCH_SIZE_PHASE_1 = 32
    EPOCHS_PHASE_1 = 15
    LR_PHASE_1 = 1e-4

    # Phase 2: Fine-Tuning
    # Focus on fine-grained disease artifacts with high resolution and lower LR
    IMG_SIZE_PHASE_2 = 512
    BATCH_SIZE_PHASE_2 = 16  # Reduced batch size for larger images to fit in GPU memory
    EPOCHS_PHASE_2 = 15
    LR_PHASE_2 = 1e-5

    # ==========================================
    # Inference Configuration
    # ==========================================
    # Inference is performed at the resolution of the final training phase
    INFERENCE_IMG_SIZE = 512
    USE_TTA = True  # Test Time Augmentation (Horizontal Flip)
