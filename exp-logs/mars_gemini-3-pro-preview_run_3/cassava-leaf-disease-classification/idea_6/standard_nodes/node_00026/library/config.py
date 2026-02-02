import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "Cassava_Dual_Stream_ViT_EffNet"
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_6"
    OUTPUT_DIR = WORKING_DIR  # Alias

    # Model Checkpoint Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Dual-Stream Architecture
    MODEL_NAME_A = "vit_base_patch16_384"  # Stream A: Global Context
    MODEL_NAME_B = "tf_efficientnet_b4"  # Stream B: Local Detail

    NUM_CLASSES = 5
    IMG_SIZE = 384  # Unified resolution for both backbones
    DROPOUT_RATE = 0.3  # Dropout for the fusion head

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 10
    BATCH_SIZE = 16  # Physical batch size per step
    ACCUMULATION_STEPS = (
        2  # Gradient accumulation to achieve effective batch size of 32
    )

    LEARNING_RATE = 1e-4  # Fine-tuning LR
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Gradient clipping

    # Loss Function
    LABEL_SMOOTHING = 0.1

    # Scheduler (Cosine Annealing)
    T_MAX = 10  # Cycles match epochs
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 3

    # =========================================================================
    # Augmentation & Regularization
    # =========================================================================
    # MixUp / CutMix
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    MIX_PROB = 0.5  # Probability of applying MixUp/CutMix

    # Test Time Augmentation (TTA)
    USE_TTA = True
    TTA_STEPS = 3  # Original + Horizontal Flip + Vertical Flip

    @classmethod
    def setup(cls):
        """
        Create necessary directories for output and submission.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration.
        """
        print(f"\n{'='*20} CONFIGURATION {'='*20}")
        print(f"Device: {cls.DEVICE}")
        print(f"Model A: {cls.MODEL_NAME_A}")
        print(f"Model B: {cls.MODEL_NAME_B}")
        print(f"Resolution: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE} (Accum: {cls.ACCUMULATION_STEPS})")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"{'='*55}\n")
