import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Idea 7: Hybrid Router-Generator Network.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging: Set to an integer (e.g., 10000) to limit dataset size for fast checking.
    # Set to None to use the full dataset.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Input Files (Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output/Checkpoint Paths
    ROUTER_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "router_checkpoints")
    GENERATOR_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "generator_checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths for Processed Data
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # ==========================================
    # Model Architecture
    # ==========================================
    # Router: Transformer Encoder for Sequence Labeling (Token Classification)
    ROUTER_MODEL_NAME = "roberta-base"

    # Generator: Character-level Seq2Seq for complex normalization
    # ByT5 is robust to character-level noise and requires no tokenization
    GENERATOR_MODEL_NAME = "google/byt5-small"

    # ==========================================
    # Hyperparameters
    # ==========================================
    # Router Params
    ROUTER_MAX_LEN = 128
    ROUTER_TRAIN_BATCH_SIZE = 32
    ROUTER_VAL_BATCH_SIZE = 64
    ROUTER_LR = 2e-5
    ROUTER_EPOCHS = 2  # Sufficient for convergence on this task size
    ROUTER_WEIGHT_DECAY = 0.01

    # Generator Params
    GEN_MAX_INPUT_LEN = 128
    GEN_MAX_TARGET_LEN = 128
    GEN_TRAIN_BATCH_SIZE = 32
    GEN_VAL_BATCH_SIZE = 64
    GEN_LR = 3e-4
    GEN_EPOCHS = 3
    GEN_WEIGHT_DECAY = 0.01

    # ==========================================
    # Class Definitions & Routing Logic
    # ==========================================
    # Full list of classes expected in the dataset
    CLASSES = [
        "PLAIN",
        "PUNCT",
        "DATE",
        "LETTERS",
        "CARDINAL",
        "VERBATIM",
        "DECIMAL",
        "MEASURE",
        "MONEY",
        "ORDINAL",
        "DIGIT",
        "ELECTRONIC",
        "TELEPHONE",
        "TIME",
        "FRACTION",
        "ADDRESS",
    ]

    # Mappings
    CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
    ID2CLASS = {i: c for i, c in enumerate(CLASSES)}
    NUM_CLASSES = len(CLASSES)

    # Routing Sets
    # 1. Structured Classes: Handled by Deterministic Rules (Python Functions)
    #    These classes have strict grammars where hallucination is unacceptable.
    STRUCTURED_CLASSES = {
        "PLAIN",
        "PUNCT",
        "CARDINAL",
        "ORDINAL",
        "DIGIT",
        "LETTERS",
        "MONEY",
        "DECIMAL",
        "FRACTION",
    }

    # 2. Unstructured/Complex Classes: Handled by Neural Generator (ByT5)
    #    These classes have high variance or ambiguity (e.g., "St." -> Street or Saint).
    UNSTRUCTURED_CLASSES = {
        "DATE",
        "VERBATIM",
        "MEASURE",
        "ELECTRONIC",
        "TELEPHONE",
        "TIME",
        "ADDRESS",
    }

    @staticmethod
    def setup():
        """Creates necessary directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.ROUTER_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.GENERATOR_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize setup immediately when imported to ensure directories exist
Config.setup()
