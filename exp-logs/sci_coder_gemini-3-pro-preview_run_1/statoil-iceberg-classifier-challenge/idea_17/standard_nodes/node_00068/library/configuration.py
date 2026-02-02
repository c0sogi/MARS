import os
import torch


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"

    # Specific file paths
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Original image size is 75x75
    ORIGINAL_IMG_SIZE = 75
    # Upsample to 224x224 for ResNet
    IMG_SIZE = 224
    # Number of channels (Band 1, Band 2, Average)
    IN_CHANNELS = 3

    # Data Loading
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "resnet18"
    PRETRAINED = True
    # Dimension of the feature vector after Global Average Pooling for ResNet18
    BACKBONE_OUT_DIM = 512
    # Dropout rate for the classification head
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Phase 1: Calibration (Finding Convergence Epoch)
    MAX_EPOCHS_PHASE_1 = 50
    EARLY_STOPPING_PATIENCE = 10

    # Optimizer (AdamW)
    # Lower LR and Weight Decay for stable fine-tuning (Cite Lesson 00067, 00009)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3

    # Loss Functions
    LABEL_SMOOTHING = 0.05

    # =========================================================================
    # SWA (Stochastic Weight Averaging) Parameters
    # =========================================================================
    # Phase 2: Production
    # Number of epochs to run SWA after the convergence epoch
    SWA_EPOCHS = 12
    # Constant learning rate for SWA phase
    SWA_LR = 1e-4

    # =========================================================================
    # Inference / TTA
    # =========================================================================
    # Number of full-fit models to train for the ensemble
    NUM_ENSEMBLE_MODELS = 5

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        print("=" * 30)
        print("CONFIGURATION")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
