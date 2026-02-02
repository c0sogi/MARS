import os
import torch


class Config:
    """
    Configuration for Diversity-Augmented ResNet34-d Ensemble Distillation with SWA.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SUBSET_SIZE = 50  # Number of samples if DEBUG is True

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Source data
    SPECTROGRAM_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "spectrograms")
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching and checkpoints
    WORKING_DIR = "./working/idea_24"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    NUM_CLASSES = 19

    # High-Fidelity Resolution: 256 (Freq) x 640 (Time)
    IMG_HEIGHT = 256
    IMG_WIDTH = 640
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # ImageNet Normalization Constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Structural Innovation: ResNet34-d (Deep stem)
    BACKBONE = "resnet34d"
    PRETRAINED = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Training Configuration
    # =========================================================================
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # =========================================================================
    # Stage 1: Diversity-Augmented Teacher Ensemble
    # =========================================================================
    NUM_TEACHERS = 3

    # Augmentation Heterogeneity: Different Mixup intensities for each teacher
    # Teacher 1: Conservative (0.1)
    # Teacher 2: Balanced (0.2)
    # Teacher 3: Strong (0.3)
    TEACHER_MIXUP_ALPHAS = [0.1, 0.2, 0.3]

    # SWA Protocol for Teachers: Active in final 25% of epochs
    # 50 * 0.75 = 37.5 -> Start at epoch 37
    TEACHER_SWA_START_EPOCH = int(EPOCHS * 0.75)
    TEACHER_SWA_LR = 1e-4

    # =========================================================================
    # Stage 2 & 3: Distillation & Student Training
    # =========================================================================
    # Student Mixup Intensity
    STUDENT_MIXUP_ALPHA = 0.2

    # SWA Protocol for Student: Active in final 30% of epochs
    # 50 * 0.70 = 35 -> Start at epoch 35
    STUDENT_SWA_START_EPOCH = int(EPOCHS * 0.70)
    STUDENT_SWA_LR = 1e-4

    # Pseudo-labeling
    PSEUDO_LABEL_PATH = os.path.join(WORKING_DIR, "pseudo_labels.parquet")

    # =========================================================================
    # Inference
    # =========================================================================
    TTA_FLIP = True  # Use Horizontal Flip TTA
