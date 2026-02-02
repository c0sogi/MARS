import os
import torch
import numpy as np
import random


class Config:
    def __init__(self):
        # ==========================================
        # 1. Paths and Directories
        # ==========================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_8"

        # Metadata Files
        self.TRAIN_FILE = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_FILE = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_FILE = os.path.join(self.METADATA_DIR, "test.csv")

        # Output Directories
        self.CHECKPOINT_DIR = os.path.join(self.WORKING_DIR, "checkpoints")
        self.CACHE_DIR = os.path.join(self.WORKING_DIR, "cache")
        self.SUBMISSION_DIR = "./submission"

        # Create necessary directories
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # ==========================================
        # 2. Model Checkpoints
        # ==========================================
        # Router: Contextual Classifier
        self.ROUTER_MODEL_NAME = "microsoft/deberta-v3-base"

        # Generator: Context-Aware Seq2Seq
        self.GENERATOR_MODEL_NAME = "google/byt5-small"

        # ==========================================
        # 3. Semiotic Class Taxonomy
        # ==========================================
        # Path A: Deterministic Rule Engine (Rigid classes)
        # Note: 'DECIMAL' and 'DIGIT' are often treated similarly; placing DECIMAL in rules.
        self.RULE_BASED_CLASSES = {
            "PLAIN",
            "PUNCT",
            "CARDINAL",
            "ORDINAL",
            "DIGIT",
            "LETTERS",
            "DECIMAL",
        }

        # Path B: Context-Aware Neural Generator (Ambiguous/Complex classes)
        self.NEURAL_BASED_CLASSES = {
            "DATE",
            "TIME",
            "MONEY",
            "MEASURE",
            "ADDRESS",
            "TELEPHONE",
            "ELECTRONIC",
            "VERBATIM",
        }

        # Full set of classes (for Router classification head)
        self.ALL_CLASSES = sorted(
            list(self.RULE_BASED_CLASSES.union(self.NEURAL_BASED_CLASSES))
        )
        self.NUM_CLASSES = len(self.ALL_CLASSES)
        self.CLASS2ID = {c: i for i, c in enumerate(self.ALL_CLASSES)}
        self.ID2CLASS = {i: c for i, c in enumerate(self.ALL_CLASSES)}

        # ==========================================
        # 4. Hyperparameters
        # ==========================================
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Data Processing
        self.MAX_LENGTH_ROUTER = 256  # Max sentence length for DeBERTa
        self.MAX_LENGTH_GENERATOR = 128  # Max sequence length for ByT5
        self.CONTEXT_WINDOW = 3  # +/- tokens for generator context

        # Training - Router
        self.ROUTER_BATCH_SIZE = 8
        self.ROUTER_LEARNING_RATE = 2e-5
        self.ROUTER_EPOCHS = 3
        self.ROUTER_PATIENCE = 1  # Early stopping patience

        # Training - Generator
        self.GENERATOR_BATCH_SIZE = 8
        self.GENERATOR_LEARNING_RATE = 1e-4
        self.GENERATOR_EPOCHS = 5
        self.GENERATOR_PATIENCE = 1

        # Sampling for Router Training
        # Keep 100% of non-PLAIN sentences
        # Keep 1% of PLAIN-only sentences (downsampling)
        self.PLAIN_DOWNSAMPLE_RATIO = 0.01

    def seed_everything(self):
        """Sets the random seed for reproducibility."""
        random.seed(self.SEED)
        os.environ["PYTHONHASHSEED"] = str(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Instantiate config to be imported by other modules
cfg = Config()
cfg.seed_everything()
