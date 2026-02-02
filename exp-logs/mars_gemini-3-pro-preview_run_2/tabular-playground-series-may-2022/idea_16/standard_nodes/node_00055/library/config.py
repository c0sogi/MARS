import os
import torch
import numpy as np
import random


class Config:
    # --------------------------------------------------------------------------
    # File System Paths
    # --------------------------------------------------------------------------
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Raw Data Paths
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    PROCESSED_DATA = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Dimensions & Processing
    # --------------------------------------------------------------------------
    # 30 Continuous features (f_00 to f_30, excluding f_27)
    NUM_CONT_FEATURES = 30
    # Sequence length for f_27 (string decomposed into chars)
    SEQ_LEN = 10
    # Vocabulary size for characters (A-Z + padding)
    VOCAB_SIZE = 27

    # --------------------------------------------------------------------------
    # Model Architecture: Gated-Transformer ResFunnel Hybrid
    # --------------------------------------------------------------------------
    # Stream 1: Gated Transformer
    EMBED_DIM = 32
    NUM_TRANSFORMER_LAYERS = 2
    NUM_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    # Gated FFN: UpProj to 8x, GLU reduces to 4x hidden
    FFN_EXPANSION_FACTOR = 8

    # Stream 2: ResFunnel Backbone
    BACKBONE_STAGES = [512, 256, 128]
    BACKBONE_DROPOUT = 0.35

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Aggressive Step Decay)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Early Stopping
    PATIENCE = 5

    # --------------------------------------------------------------------------
    # System & Debugging
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Quick Run
    DEBUG = False
    DEBUG_SAMPLES = 10000

    @staticmethod
    def setup():
        """
        Initializes the environment:
        1. Creates necessary writable directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        seed = Config.SEED
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration loaded. Device: {Config.DEVICE}, Seed: {Config.SEED}")
