import os
import torch


class Config:
    """
    Centralized configuration for the Glioblastoma MGMT detection pipeline.
    Implements the 'Asymmetric Grouped EfficientNet' strategy settings.
    """

    # --------------------------------------------------------------------------
    # 1. File System & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working & Output
    WORKING_DIR = "./working"
    # Specific cache directory for deterministic data processing (Idea 40)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_40")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Hyperparameters
    # --------------------------------------------------------------------------
    # Image Geometry
    IMG_SIZE = 224

    # Modalities and Channels
    # 4 Modalities: FLAIR, T1w, T1wCE, T2w
    # 3 Slices per modality -> 12 Input Channels total
    MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
    SLICES_PER_MODALITY = 3
    TOTAL_CHANNELS = len(MODALITIES) * SLICES_PER_MODALITY  # 12

    # ROI Selection (Fidelity-Aligned)
    ROI_START_PCT = 0.15
    ROI_END_PCT = 0.85

    # Stochastic Multi-Scale Stacking
    # Randomly choose between Stride 2 (Texture) and Stride 5 (Context) during training
    STRIDE_OPTIONS = [2, 5]

    # Debugging / Development
    # Set to an integer (e.g., 50) to limit dataset size for rapid prototyping
    DEBUG_DATA_LIMIT = None

    # Exclusions (Problematic cases identified in task description)
    EXCLUDE_IDS = [109, 123, 709]

    # --------------------------------------------------------------------------
    # 3. Model & Training Hyperparameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"

    # Optimization
    SEED = 42
    BATCH_SIZE = 32  # Optimized for A100 GPU
    EPOCHS = 20  # Sufficient for convergence with Early Stopping
    LR = 1e-4  # Low Learning Rate
    WEIGHT_DECAY = 1e-2  # Aggressive Weight Decay
    DROPOUT_RATE = 0.5  # Regularization for Classification Head

    # Scheduler & Stopping
    EARLY_STOPPING_PATIENCE = 5

    # Hardware
    NUM_WORKERS = 4  # 12 vCPUs available, 4 is a safe efficient number
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducibility
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
        print(f"Config setup complete. Device: {cls.DEVICE}, Cache: {cls.CACHE_DIR}")
