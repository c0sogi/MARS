import os
import torch


class Config:
    # General Setup
    SEED = 42
    N_FOLDS = 5
    DEBUG = False  # Set to True for fast debugging runs
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # Infrastructure
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # File Paths
    # Using metadata paths as per instructions for pre-stratified splits
    TRAIN_META_PATH = "./metadata/train.csv"
    VAL_META_PATH = "./metadata/val.csv"
    TEST_META_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Artifacts and Caching
    # All outputs (models, cache) go here
    ARTIFACTS_DIR = "./working/idea_12/"
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Architecture
    # Heterogeneous Ensemble: DeBERTa-v3 (Relative Pos) + RoBERTa (Absolute Pos)
    # This diversity helps resolve boundary ambiguity.
    MODEL_BACKBONES = ["microsoft/deberta-v3-large", "roberta-large"]

    # Training Hyperparameters
    MAX_LEN = 128  # Sufficient for tweets
    TRAIN_BATCH_SIZE = 8  # Adjusted for 40GB GPU + Large Models
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Optimization & Loss
    USE_AMP = True  # Mixed Precision is mandatory for efficiency
    LABEL_SMOOTHING = 0.1  # To handle noisy manual annotations
    EARLY_STOPPING_PATIENCE = 2

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed(cls.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize environment immediately upon import
Config.setup()
