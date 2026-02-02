import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Configuration class for the Multi-Stage Domain-Adapted Stacked Ensemble.
    """

    # === Experiment Setup ===
    SEED = 42
    EXP_NAME = "idea_8"
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # === Paths ===
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{EXP_NAME}"

    # Input Data (Using Metadata as requested)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    MLM_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "mlm_checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # === Model Architecture ===
    MODEL_BACKBONE = "microsoft/deberta-v3-large"
    MAX_LENGTH = 1024  # Long context to capture full essay argumentation
    NUM_LABELS = 1  # Regression output (score)
    GRADIENT_CHECKPOINTING = True  # Essential for Large model + 1024 seq len

    # === Stage 1: MLM Pre-training ===
    MLM_EPOCHS = 3
    MLM_BATCH_SIZE = 2
    MLM_LEARNING_RATE = 2e-5
    MLM_MASK_PROB = 0.15

    # === Stage 2: Supervised Fine-Tuning ===
    NUM_FOLDS = 5
    EPOCHS = 4
    # Batch size is kept low to fit 1024 tokens on A100 GPU
    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 8  # Effective batch size = 2 * 8 = 16

    # Optimization
    LEARNING_RATE = 1e-5  # Lower learning rate for the backbone
    HEAD_LEARNING_RATE = 1e-3  # Higher learning rate for the linear head
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 10.0
    WARMUP_RATIO = 0.1
    SCHEDULER_TYPE = "cosine"

    # Layer-wise Learning Rate Decay (LLRD)
    # Stabilizes training of large backbones by decaying LR for lower layers
    LLRD_DECAY = 0.9

    # === Stage 3: Stacking (Meta-Model) ===
    META_MODEL_TYPE = "ridge"  # Options: 'ridge', 'lgbm'

    # === Hardware & Performance ===
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup_directories(cls):
        """
        Creates the necessary directory structure for the experiment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.MLM_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
