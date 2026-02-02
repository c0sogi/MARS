import os
import torch


class Config:
    """
    Centralized configuration for the Glioblastoma Subtype Prediction task.
    Implements settings for 'Asymmetric Grouped EfficientNet with Logical-Consensus ROI Pipeline'.
    """

    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    METADATA_DIR = "./metadata"

    # Working Directory for Caching and Model Checkpoints
    # Requirement: Ensure ./working/idea_33/ exists
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_33")
    CACHE_DIR = IDEA_DIR
    MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Create necessary directories immediately
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Settings
    # --------------------------------------------------------------------------
    IMG_SIZE = (224, 224)

    # ROI Selection Logic (Logical-Consensus)
    ROI_MODALITIES = ["FLAIR", "T1wCE"]  # Modalities used to calculate consensus peak
    ROI_DEPTH_RANGE = (0.15, 0.85)  # Exclude top/bottom 15% of volume

    # Stacking Logic
    INPUT_MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
    NUM_SLICES_PER_MODALITY = 3
    STRIDE = 5
    STACK_OFFSETS = [-5, 0, 5]  # Relative to anchor slice

    # --------------------------------------------------------------------------
    # 3. Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"

    # Input Dimensions: 4 Modalities * 3 Slices = 12 Channels
    INPUT_CHANNELS = len(INPUT_MODALITIES) * NUM_SLICES_PER_MODALITY

    # Stem Modification
    STEM_GROUPS = 4  # Independent processing for each modality group in the first layer

    # Head Modification
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5

    # --------------------------------------------------------------------------
    # 4. Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 15

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # 5. Augmentation & Inference
    # --------------------------------------------------------------------------
    # Augmentation
    AUG_ROTATION = 15  # Degrees (+/-)

    # Test Time Augmentation
    TTA_FLIPS = ["none", "horizontal", "vertical"]
