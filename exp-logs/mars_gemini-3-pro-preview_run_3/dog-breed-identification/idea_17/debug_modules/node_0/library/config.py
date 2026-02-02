import os
import torch


class Config:
    """
    Configuration for Dog Breed Classification Task.
    Implements strategy: Hierarchical Stratified Ensemble with Corrected Transfer Dynamics.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"

    # Metadata paths (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Configuration
    # ==========================================
    # Architecture: ConvNeXt Small
    # Weights: ImageNet-21k pre-training fine-tuned on ImageNet-1k
    # This provides better feature separation for fine-grained tasks than standard 1k weights.
    MODEL_NAME = "convnext_small.in12k_ft_in1k"
    NUM_CLASSES = 120

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 224
    BATCH_SIZE = 64  # Optimized for A100 40GB to ensure stability
    NUM_WORKERS = 4

    # Normalization constants (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Training Configuration
    # ==========================================
    N_FOLDS = 5
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase 1: Linear Probing / Head Warmup
    # High LR to quickly align the random head with frozen backbone features
    EPOCHS_WARMUP = 1
    LR_WARMUP = 1e-3

    # Phase 2: Full Fine-Tuning
    # Low LR to gently adapt the backbone without destroying priors
    EPOCHS_FINE = 30
    LR_FINE = 1e-5

    # Optimization
    WEIGHT_DECAY = 1e-4
    SCHEDULER_MIN_LR = 1e-7

    # ==========================================
    # Ensembling Configuration
    # ==========================================
    # Number of top checkpoints (by val_loss) to average for Manual Model Soup
    SOUP_TOP_K = 3

    # ==========================================
    # Debugging
    # ==========================================
    # Set to True to train on a small subset for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def setup(cls):
        """Creates necessary working and output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
