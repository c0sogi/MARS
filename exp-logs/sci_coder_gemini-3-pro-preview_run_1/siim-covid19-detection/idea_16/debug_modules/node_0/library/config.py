import os
import torch


class Config:
    """
    Configuration for Anti-Aliased ResNet18 Multi-Task U-Net with EMA.
    Idea ID: idea_16
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    PROJECT_NAME = "AntiAliasedResNet18_UNet_EMA"
    IDEA_ID = "idea_16"
    DEBUG = False  # Set to True to run on a small subset for testing

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_ID)
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Artifact Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Resize all images to 512x512 pixels
    IMAGE_SIZE = (512, 512)

    # Class Definitions
    STUDY_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    NUM_CLASSES_STUDY = len(STUDY_LABELS)
    NUM_CLASSES_IMAGE = 1  # 'opacity'

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "resnet18"
    USE_ANTI_ALIASING = True  # Use BlurPool in backbone
    PRETRAINED = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Access to 1 NVIDIA A100-SXM4-40GB GPU allows for reasonable batch sizes
    BATCH_SIZE = 16
    NUM_EPOCHS = 20

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    OPTIMIZER_NAME = "AdamW"
    SCHEDULER_NAME = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # Loss Weighting (1:10 ratio)
    # Prioritizes dense prediction to force shared encoder to learn spatial features
    LOSS_WEIGHT_CLS = 1.0
    LOSS_WEIGHT_SEG = 10.0

    # Exponential Moving Average (EMA)
    USE_EMA = True
    EMA_DECAY = 0.999

    # -------------------------------------------------------------------------
    # Augmentation & Preprocessing
    # -------------------------------------------------------------------------
    # Consistency Constraint: If opacity is occluded by CoarseDropout,
    # mask_fill_value=0 removes it from the ground truth mask.
    MASK_FILL_VALUE = 0

    # -------------------------------------------------------------------------
    # Inference Strategy
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # 12 vCPUs available

    # Test-Time Augmentation
    TTA_FLIP = True

    # Post-processing
    # Logical Gating: If study is 'Negative', force image prediction to 'none'
    GATED_PREDICTION = True

    @classmethod
    def display(cls):
        """Prints the current configuration."""
        print(f"\n{'='*20} CONFIGURATION {'='*20}")
        print(f"Idea ID      : {cls.IDEA_ID}")
        print(f"Device       : {cls.DEVICE}")
        print(f"Image Size   : {cls.IMAGE_SIZE}")
        print(f"Batch Size   : {cls.BATCH_SIZE}")
        print(f"Epochs       : {cls.NUM_EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Loss Weights : Cls={cls.LOSS_WEIGHT_CLS}, Seg={cls.LOSS_WEIGHT_SEG}")
        print(f"Anti-Aliasing: {cls.USE_ANTI_ALIASING}")
        print(f"Use EMA      : {cls.USE_EMA}")
        print(f"Working Dir  : {cls.WORKING_DIR}")
        print(f"{'='*55}\n")
