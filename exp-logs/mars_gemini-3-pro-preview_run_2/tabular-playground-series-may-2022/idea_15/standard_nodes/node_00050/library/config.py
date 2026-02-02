import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # File System Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Hyperparameters
    # --------------------------------------------------------------------------
    # f_00 to f_30 are continuous, except f_27 which is categorical.
    # Typically 30 continuous features in this dataset.
    NUM_CONTINUOUS_FEATURES = 30

    # f_27 is a string of length 10
    SEQUENCE_LENGTH = 10

    # Characters are typically A-Z (26 unique tokens).
    # We add 1 for potential padding or 0-indexing offset if needed.
    VOCAB_SIZE = 27

    # --------------------------------------------------------------------------
    # Model Architecture (Gated Transformer-ResFunnel Hybrid)
    # --------------------------------------------------------------------------
    # Stream 1: Gated Transformer
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_FFN_DIM = 128  # Internal dim for Gated FFN
    TRANSFORMER_DROPOUT = 0.1  # Low dropout for attention mechanism

    # Stream 2 & Backbone: ResFunnel-GLU
    # Fusion happens by concatenating flattened transformer out + raw continuous
    BACKBONE_WIDTHS = [512, 256, 128]
    BACKBONE_DROPOUT = 0.35  # High dropout for dense layers

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 1024
    EPOCHS = 40  # 35+ requested
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Strong regularization

    # Scheduler: Aggressive Step Decay
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Ensures the working directory exists.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
