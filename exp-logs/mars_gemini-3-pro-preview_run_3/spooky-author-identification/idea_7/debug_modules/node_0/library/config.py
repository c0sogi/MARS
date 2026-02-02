import os
import torch
import random
import numpy as np


class Config:
    """
    Global configuration for the Transductive Knowledge Distillation pipeline.
    """

    # --- General Settings ---
    PROJECT_NAME = "author_id_idea_7"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # --- Labels ---
    # Mapping based on dataset description
    LABEL2ID = {"EAP": 0, "HPL": 1, "MWS": 2}
    ID2LABEL = {0: "EAP", 1: "HPL", 2: "MWS"}
    NUM_CLASSES = 3

    # --- Hardware ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # --- Data Paths ---
    # Using metadata files as primary source
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --- Artifact Storage (Cache) ---
    # Directory for storing processed features, models, and temporary files
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")

    # --- Model Configuration ---
    # Backbones for the ensemble
    MODEL_BACKBONES = ["microsoft/deberta-v3-base", "roberta-base"]

    # Text Processing
    MAX_LENGTH = 256  # Covers majority of sentences without excessive padding

    # --- Training Hyperparameters ---
    # General
    N_FOLDS = 5

    # Masked Language Modeling (DAPT)
    MLM_EPOCHS = 5
    MLM_BATCH_SIZE = 16
    MLM_LR = 2e-5
    MLM_WEIGHT_DECAY = 0.01
    MLM_MASK_PROB = 0.15

    # Supervised Fine-Tuning & Distillation
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32

    FT_EPOCHS = 5
    FT_LR = 2e-5
    FT_WEIGHT_DECAY = 0.01
    FT_WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 2

    # --- Statistical Branch Configuration ---
    # TF-IDF Vectorizer settings
    TFIDF_WORD_PARAMS = {
        "analyzer": "word",
        "token_pattern": r"\w{1,}",
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }

    TFIDF_CHAR_PARAMS = {
        "analyzer": "char",
        "ngram_range": (3, 5),
        "max_features": 25000,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }

    # --- Distillation Configuration ---
    DISTILLATION_TEMP = 2.0
    # Weight for soft targets (KL Div) vs hard labels (Cross Entropy)
    # Loss = (1 - ALPHA) * CE + ALPHA * KL
    DISTILLATION_ALPHA = 0.5

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets seeds."""
        for directory in [
            cls.WORKING_DIR,
            cls.SUBMISSION_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.PREDICTIONS_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

        cls.set_seed(cls.SEED)


# Initialize environment immediately upon import
Config.setup()
