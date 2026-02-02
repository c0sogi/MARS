import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Global Settings & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    IDEA_NAME = "idea_35"

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = "./submission"

    # Raw Data Sources
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Splits)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    # Features
    NUM_CONTINUOUS_FEATURES = 30  # f_00 to f_30 (excluding f_27)
    SEQUENCE_LENGTH = 10  # Length of decomposed f_27 string
    VOCAB_SIZE = 26  # Unique characters (A-Z)

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Stream 1: Categorical (Post-Norm Transformer)
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "gelu"

    # Stream 2: Continuous (Raw Normalized)
    # (No specific params, just input dimension matches NUM_CONTINUOUS_FEATURES)

    # Backbone: LayerScaled SwiGLU ResFunnel
    # Stages: 512 -> 256 -> 128
    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3
    LAYERSCALE_INIT = 1e-5
    STOCHASTIC_DEPTH_MAX_RATE = 0.2
    BACKBONE_DROPOUT = 0.35

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 40
    LEARNING_RATE = 1e-3

    # Optimization
    WEIGHT_DECAY = 1e-2
    # Specific parameter groups (biases, layernorm, etc.) will have 0.0 decay in logic

    # Scheduler (Step Decay)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
