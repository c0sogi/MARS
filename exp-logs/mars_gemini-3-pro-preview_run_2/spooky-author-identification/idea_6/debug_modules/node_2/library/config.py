import os
import torch


class Config:
    """
    Global configuration for the Author Identification pipeline.
    Defines hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use when DEBUG=True

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Compute Environment
    # ==========================================
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Data Configuration
    # ==========================================
    CLASSES = ["EAP", "HPL", "MWS"]
    NUM_LABELS = 3
    N_FOLDS = 5  # Stratified 5-Fold Cross-Validation

    # ==========================================
    # Neural Model Hyperparameters (DeBERTa)
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 85

    # Training Dynamics
    # A100 40GB allows reasonable batch sizes, but DeBERTa-Large is heavy.
    # We use Gradient Accumulation to reach effective batch size of 16.
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4  # 4 * 4 = 16 effective batch size

    # Optimization
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    PATIENCE = 1  # Aggressive early stopping as per strategy

    # ==========================================
    # Classical Models & Feature Engineering
    # ==========================================
    # TF-IDF Settings
    NGRAM_RANGE_WORD = (1, 3)
    NGRAM_RANGE_CHAR = (2, 5)
    MIN_DF = 2  # Prune singleton noise

    # SVD Settings for XGBoost
    SVD_COMPONENTS = 100

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to ensure environment is ready
Config.setup()
