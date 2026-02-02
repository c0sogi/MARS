import os
import torch


class Config:
    """
    Global configuration for the Asymmetric Deep Parallel Vector-DCN-ResNet experiment.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    # Input Metadata Paths (Read-only)
    TRAIN_DATA_PATH = "./metadata/train.parquet"
    VAL_DATA_PATH = "./metadata/val.parquet"
    TEST_DATA_PATH = "./metadata/test.parquet"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working Directory for Caching and Artifacts
    # Requirement: Use ./working/idea_32/
    CACHE_DIR = "./working/idea_32/"

    # Model Checkpoint Path
    # We save the best model here during training
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Configuration
    # ==========================================
    TARGET_COL = "Cover_Type"
    ID_COL = "Id"

    # The dataset has 7 classes (integers 1-7).
    # We typically map these to 0-6 for CrossEntropyLoss.
    NUM_CLASSES = 7

    # Feature Engineering Flags
    ADD_ASPECT_CYCLICAL = True
    ADD_HYDROLOGY_EUCLIDEAN = True
    ADD_ABSOLUTE_HYDROLOGY = True
    ADD_AMENITIES_MEAN = True

    # Debugging: Set to an integer (e.g., 10000) to subsample data for rapid testing.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    # Branch 1: Asymmetric Vector-Based DCN
    # Explicitly limited to 3 layers to decouple interaction order from depth
    DCN_LAYERS = 3

    # Branch 2: Deep Full Pre-Activation ResNet Backbone
    RESNET_BLOCKS = 4  # Depth
    HIDDEN_DIM = 512  # Width

    # Regularization
    DROPOUT_RATE = 0.2

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42

    # Budgeting
    BATCH_SIZE = 4096
    EPOCHS = 60

    # Optimizer: AdamW
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler: ReduceLROnPlateau (Aggressive)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 3
    SCHEDULER_MODE = "max"  # We monitor validation accuracy
    SCHEDULER_MIN_LR = 1e-7

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 8

    # ==========================================
    # 5. Hardware & System
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Number of data loading workers
    NUM_WORKERS = 4

    # Determinism settings
    # We disable strict determinism for performance as per Lesson 00070
    CUDNN_DETERMINISTIC = False
    CUDNN_BENCHMARK = True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for cache and submission.
        Should be called at the start of the execution.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories on import
Config.setup()
