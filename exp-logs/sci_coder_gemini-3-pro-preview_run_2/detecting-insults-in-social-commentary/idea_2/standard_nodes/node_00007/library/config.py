import torch
import os


class Config:
    # Reproducibility
    SEED = 42

    # Model Configuration
    MODEL_NAME = "roberta-base"
    MAX_LEN = 128
    LOWERCASE = (
        False  # RoBERTa is case-sensitive usually, but DistilRoBERTa base is cased.
    )

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 75  # ~10% of training steps

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    WORKING_DIR = "./working/idea_3"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
