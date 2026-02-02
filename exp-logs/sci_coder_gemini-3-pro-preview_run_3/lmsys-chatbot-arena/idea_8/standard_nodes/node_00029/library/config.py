import os
import torch


class Config:
    # ==== General Settings ====
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # ==== File Paths ====
    # Metadata paths (Input)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output paths
    # Using idea_8 as the specific working directory for this iteration
    WORKING_DIR = "./working/idea_8"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==== Model Architecture ====
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 512
    NUM_CLASSES = 3  # Winner A, Winner B, Tie
    DROPOUT = 0.1

    # Pooling Strategy
    N_LAST_LAYERS_POOLING = 4  # Number of last layers to pool from

    # ==== Training Hyperparameters ====
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 10.0
    ACCUMULATION_STEPS = 1
    WARMUP_RATIO = 0.1
    USE_FP16 = True

    # Early Stopping
    PATIENCE = 2

    # ==== Data Processing ====
    # Augmentation: Swap A and B to double dataset size
    USE_SYMMETRIC_AUGMENTATION = True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for outputs and cache.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducibility
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)


# Initialize directories immediately upon import
Config.setup()
