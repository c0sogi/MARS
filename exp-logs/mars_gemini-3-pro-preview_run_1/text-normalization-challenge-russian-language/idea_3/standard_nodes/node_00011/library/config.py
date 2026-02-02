import os
import random
import numpy as np
import torch
import hashlib
import json


class Config:
    # --- General ---
    SEED = 42
    IDEA_DIR = "./working/idea_3"

    # --- Data Paths ---
    TRAIN_META_PATH = "./metadata/train.csv"
    VAL_META_PATH = "./metadata/val.csv"
    TEST_META_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/ru_sample_submission_2.csv"

    # --- N-Gram / Symbolic Layer ---
    NGRAM_ORDER = 3  # Trigram -> Bigram -> Unigram

    # --- Neural Model (Character-Level Transformer) ---
    # Vocabulary: ~200 chars for Cyrillic/Latin/Symbols + Special tokens
    VOCAB_SIZE = 300
    MAX_SEQ_LEN = 128  # Max length for char sequence (context + target)

    # Model Architecture
    D_MODEL = 256
    NHEAD = 4
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # --- Training ---
    BATCH_SIZE = 1024
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5
    NUM_WORKERS = 8

    # --- Context Window ---
    # How many words to the left and right to include in the neural input
    CONTEXT_WINDOW = 2

    @classmethod
    def get_artifacts_dir(cls):
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        return cls.IDEA_DIR

    @classmethod
    def get_config_dict(cls):
        """Returns a dictionary of hyperparameters that affect the model logic."""
        return {
            "ngram_order": cls.NGRAM_ORDER,
            "vocab_size": cls.VOCAB_SIZE,
            "max_seq_len": cls.MAX_SEQ_LEN,
            "d_model": cls.D_MODEL,
            "nhead": cls.NHEAD,
            "num_encoder_layers": cls.NUM_ENCODER_LAYERS,
            "num_decoder_layers": cls.NUM_DECODER_LAYERS,
            "dim_feedforward": cls.DIM_FEEDFORWARD,
            "context_window": cls.CONTEXT_WINDOW,
            "seed": cls.SEED,
        }

    @classmethod
    def get_hash(cls):
        """Generates a unique hash based on the configuration."""
        config_str = json.dumps(cls.get_config_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:8]

    @classmethod
    def get_artifact_path(cls, name):
        """Returns a path for an artifact including the config hash."""
        conf_hash = cls.get_hash()
        base_name, ext = os.path.splitext(name)
        filename = f"{base_name}_{conf_hash}{ext}"
        return os.path.join(cls.get_artifacts_dir(), filename)


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
