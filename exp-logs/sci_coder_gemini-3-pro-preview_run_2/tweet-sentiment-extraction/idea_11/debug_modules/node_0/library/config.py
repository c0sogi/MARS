import os
import torch


class Config:
    """
    Centralized configuration for the Tweet Sentiment Extraction task.
    Implements the settings for a Heterogeneous Ensemble (DeBERTa-v3 + RoBERTa).
    """

    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available; setting workers to a reasonable number for data loading
    NUM_WORKERS = 4

    # ====================================================
    # Data Paths
    # ====================================================
    # Using the generated metadata files which contain the stratified splits
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output directory for saving models and cache
    OUTPUT_DIR = "./working/idea_11/"

    # ====================================================
    # Model Architecture & Ensemble Config
    # ====================================================
    # Defining the heterogeneous ensemble components
    MODEL_CONFIGS = [
        {
            "model_name": "microsoft/deberta-v3-large",
            "save_name": "deberta_v3_large",
            "batch_size": 8,  # Adjusted for A100 40GB
            "tokenizer_type": "deberta",
        },
        {
            "model_name": "roberta-large",
            "save_name": "roberta_large",
            "batch_size": 8,
            "tokenizer_type": "roberta",
        },
    ]

    # Input sequence length (Analysis showed max char len ~141, 128 tokens is safe)
    MAX_LEN = 128

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    N_FOLDS = 5
    EPOCHS = 3

    # Optimization
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.1

    # Loss Function
    LABEL_SMOOTHING = 0.1

    # Performance
    USE_FP16 = True  # Mixed Precision Training

    # ====================================================
    # Debugging & Development
    # ====================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Performs necessary setup operations like creating directories.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# Execute setup on import to ensure directories exist
Config.setup()
