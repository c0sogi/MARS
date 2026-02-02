import os
import torch


class Config:
    # Random Seed for reproducibility
    SEED = 42

    # Data Configuration
    NUM_FOLDS = 5
    NUM_CLASSES = 120
    IMG_SIZE = 224

    # Model Configuration
    # Using ConvNeXt Small pre-trained on ImageNet-22k and fine-tuned on 1k
    MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 1e-5

    # Optimization Strategy
    WARMUP_EPOCHS = 1
    SOUP_CANDIDATES = 10  # Number of final epochs to consider for Greedy Model Soup

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_12")
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs or default to 4
    NUM_WORKERS = os.cpu_count() if os.cpu_count() is not None else 4

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
