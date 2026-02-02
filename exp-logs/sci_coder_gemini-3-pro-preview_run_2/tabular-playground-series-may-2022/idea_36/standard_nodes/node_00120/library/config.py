import os
import torch
import numpy as np
import random


class Config:
    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea/experiment
    WORKING_DIR = "./working/idea_36"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Paths
    TRAIN_RAW_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_RAW_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Artifact Paths
    # Using .npz for caching processed data as requested
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Hyperparameters
    # --------------------------------------------------------------------------
    # 30 continuous features (f_00..f_30 excluding f_27)
    NUM_CONTINUOUS = 30
    # f_27 is split into 10 characters
    SEQUENCE_LENGTH = 10
    # Vocabulary size for character embeddings (A-Z + padding/unknown)
    # Safe upper bound for uppercase English letters
    VOCAB_SIZE = 40

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Transformer Stream (Post-Norm)
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "gelu"
    TRANSFORMER_NORM_FIRST = False  # Post-Norm

    # Fusion
    INIT_SEQ_SCALAR = 0.1
    INIT_CONT_SCALAR = 1.0

    # Backbone (SwiGLU ResFunnel)
    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3
    BACKBONE_DROPOUT = 0.35
    STOCHASTIC_DEPTH_MIN = 0.0
    STOCHASTIC_DEPTH_MAX = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    EPOCHS = 40
    BATCH_SIZE = 1024

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY_GROUP1 = 1e-2  # Weights
    WEIGHT_DECAY_GROUP2 = 0.0  # Biases, Norms, Scalars

    # Scheduler (Step Decay)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_reproducibility():
        """
        Sets seeds for all random number generators to ensure reproducibility.
        """
        seed = Config.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms can slow down training, but ensure exact reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
