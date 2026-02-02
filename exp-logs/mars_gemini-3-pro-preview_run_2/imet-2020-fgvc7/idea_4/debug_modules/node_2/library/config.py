import os
import torch


class Config:
    """
    Configuration for Artwork Attribute Labeling Task.
    Implements the settings for the Teacher-Student Knowledge Distillation pipeline.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    # Use A100 GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available
    NUM_WORKERS = 12

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for checkpoints and intermediate files (Idea 4)
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test")
    LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architectures (timm)
    # ==========================================
    # Teacher 1: ConvNeXt Small (CNN) - Good local features
    TEACHER_1_MODEL_NAME = "convnext_small.fb_in22k_ft_in1k_384"

    # Teacher 2: Swin Base (Transformer) - Good global context
    TEACHER_2_MODEL_NAME = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"

    # Student: ConvNeXt Base (CNN) - High capacity, efficient inference
    STUDENT_MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"

    # Task specifics
    NUM_CLASSES = 3474

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    IMAGE_SIZE = 384

    # Batch Size: A100 40GB allows ~32-48 for Swin Base 384.
    # Using 32 to be safe with AMP and overhead.
    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 64

    # Optimization
    LR_MAX = 2e-4
    WEIGHT_DECAY = 1e-4
    MIN_LR = 1e-6

    # Epochs
    # Constrained by 24h runtime.
    # Teachers need enough to be good supervisors. Student needs to absorb info.
    TEACHER_EPOCHS = 6
    STUDENT_EPOCHS = 10

    # Debugging / Development
    # Set DEBUG = True to run on a small subset (e.g., 5000 images) for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 5000

    # ==========================================
    # Loss & Distillation Settings
    # ==========================================
    # Asymmetric Loss (ASL) for Multi-Label Imbalance
    ASL_GAMMA_NEG = 4.0
    ASL_GAMMA_POS = 1.0
    ASL_CLIP = 0.05

    # Distillation Weights
    # Loss = Alpha * BCE(Student, SoftTargets) + (1 - Alpha) * ASL(Student, HardTargets)
    DISTILL_ALPHA = 0.5

    # ==========================================
    # Checkpoints & Cache
    # ==========================================
    # Saved Model Weights
    TEACHER_1_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_convnext_small.pth")
    TEACHER_2_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_swin_base.pth")
    STUDENT_CHECKPOINT = os.path.join(WORKING_DIR, "student_convnext_base.pth")

    # Cached Soft Labels (Ensemble Predictions on Train Set)
    # Format: .npy file of shape (N_train, N_classes)
    SOFT_LABELS_PATH = os.path.join(WORKING_DIR, "teacher_soft_labels.npy")

    # Validation Predictions (for Threshold Optimization)
    VAL_PREDS_PATH = os.path.join(WORKING_DIR, "val_predictions.npy")
    VAL_TARGETS_PATH = os.path.join(WORKING_DIR, "val_targets.npy")
