import os
import torch


class Config:
    """
    Configuration class for Dog Breed Classification Task.
    Implements parameters for Stratified K-Fold Ensemble with ConvNeXt-Base.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 4 (ConvNeXt-Base + K-Fold)
    WORKING_DIR = "./working/idea_4"

    # Ensure working directory exists immediately upon import
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    # We will combine Train and Val metadata for K-Fold splitting
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    NUM_CLASSES = 120

    # Geometric Processing
    # Resize to 256, then Center Crop to 224
    RESIZE_SIZE = 256
    IMAGE_SIZE = 224

    # DataLoader
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # ==========================================
    # Model Configuration
    # ==========================================
    # ConvNeXt Base pre-trained on ImageNet-22k and fine-tuned on ImageNet-1k
    MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"

    # ==========================================
    # Training Configuration
    # ==========================================
    N_FOLDS = 5

    # Phase 1: Head Adaptation (Frozen Backbone)
    EPOCHS_PHASE_1 = 3
    LR_HEAD_PHASE_1 = 1e-3

    # Phase 2: Fine-Tuning (Unfrozen Backbone with Discriminative LRs)
    EPOCHS_PHASE_2 = 12
    LR_HEAD_PHASE_2 = 1e-4
    LR_BACKBONE = 1e-6

    # Regularization
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early stopping patience

    # ==========================================
    # Inference Configuration
    # ==========================================
    USE_TTA = True  # Use Test Time Augmentation (Horizontal Flip)

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_model_path(fold_idx):
        """Returns the file path for saving/loading the model checkpoint for a specific fold."""
        return os.path.join(Config.WORKING_DIR, f"convnext_base_fold_{fold_idx}.pth")
