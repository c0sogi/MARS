import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Cache directory for deterministic data processing
    WORKING_DIR = "./working/idea_18"
    # Output path for submission
    SUBMISSION_PATH = "./submission/submission.csv"

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    # 30 continuous features (f_00 to f_30, excluding f_27)
    NUM_CONT_FEATURES = 30
    # f_27 is split into 10 characters
    SEQ_LEN = 10
    # Vocabulary size for character embeddings (A-Z + padding/unknown).
    # 26 letters + special tokens. 35 is a safe upper bound.
    VOCAB_SIZE = 35

    # --------------------------------------------------------------------------
    # Model Architecture: Multi-Scale Hybrid ResFunnel (MS-ResFunnel)
    # --------------------------------------------------------------------------
    # Stream 1: Categorical Sequence (Transformer)
    EMBEDDING_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1

    # Fusion Layer
    # Project concatenated [Flattened Transformer (32*10) + Raw Continuous (30)] -> 512
    INITIAL_WIDTH = 512

    # Stream 2 & Backbone: Pre-Activation ResFunnel
    # Stages with decreasing width
    BACKBONE_STAGES = [512, 256, 128]
    BACKBONE_DROPOUT = 0.35

    # Head: Multi-Scale Aggregation
    # Concatenates outputs of all stages: 512 + 256 + 128 = 896 input to classifier

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 40

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Aggressive StepLR)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # --------------------------------------------------------------------------
    # Compute & Debugging
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags to control dataset size for rapid testing
    DEBUG = False
    DEBUG_SAMPLES = 5000

    @classmethod
    def setup(cls):
        """
        Initialize necessary directories for caching and submission.
        """
        # Create working directory for cache
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Create submission directory
        sub_dir = os.path.dirname(cls.SUBMISSION_PATH)
        if sub_dir:
            os.makedirs(sub_dir, exist_ok=True)

        print(f"Config configured. Device: {cls.DEVICE}, Batch Size: {cls.BATCH_SIZE}")
