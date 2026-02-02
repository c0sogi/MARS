import os
import torch


class Config:
    """
    Centralized configuration for the Artwork Attribute Labeling task.
    Implements the Calibrated Heterogeneous Distillation Strategy (Idea 5).
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SUBSET_SIZE = 2000  # Number of samples to use when DEBUG is True

    # Compute resources
    # We have 12 vCPUs, setting workers to 8 leaves some overhead for main process
    NUM_WORKERS = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working directory for Idea 5
    WORKING_DIR = "./working/idea_5"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Checkpoint paths
    TEACHER_1_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_convnext_small_best.pth")
    TEACHER_2_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_swin_base_best.pth")
    STUDENT_CHECKPOINT = os.path.join(WORKING_DIR, "student_convnext_large_best.pth")

    # Soft labels storage (cached predictions from teachers)
    TEACHER_PREDS_PATH = os.path.join(WORKING_DIR, "teacher_predictions.npy")

    # Final Submission
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMG_SIZE = 384
    NUM_CLASSES = 3474

    # -------------------------------------------------------------------------
    # Model Architectures (timm)
    # -------------------------------------------------------------------------
    # Teacher 1: CNN (Local features, texture)
    TEACHER_MODEL_1 = "convnext_small_d.um_in1k"

    # Teacher 2: Transformer (Global context, composition)
    TEACHER_MODEL_2 = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"

    # Student: Large CNN (High capacity to absorb ensemble knowledge)
    STUDENT_MODEL = "convnext_large_d.um_in1k"

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch size adjusted for A100 40GB RAM and 384x384 resolution
    # Swin Base and ConvNeXt Large are memory intensive.
    TRAIN_BATCH_SIZE = 24
    VALID_BATCH_SIZE = 32

    # Training duration
    EPOCHS = 8

    # Optimization
    LEARNING_RATE = 2e-5  # Base learning rate for fine-tuning
    MAX_LR = 1e-4  # Max LR for OneCycleLR scheduler
    WEIGHT_DECAY = 0.01

    # -------------------------------------------------------------------------
    # Loss & Distillation Parameters
    # -------------------------------------------------------------------------
    # Asymmetric Loss (ASL) for handling class imbalance
    ASL_GAMMA_NEG = 4.0  # Down-weight easy negative examples
    ASL_GAMMA_POS = 0.0  # No down-weighting for positive examples
    ASL_CLIP = 0.05  # Probability margin

    # Knowledge Distillation
    # Loss = alpha * KL_Div(Student, Teacher) + (1 - alpha) * ASL(Student, Hard_Labels)
    DISTILLATION_ALPHA = 0.5
    DISTILLATION_TEMP = 4.0  # Temperature to soften probability distributions

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    TTA_FLIP = True  # Use Horizontal Flip Test-Time Augmentation
