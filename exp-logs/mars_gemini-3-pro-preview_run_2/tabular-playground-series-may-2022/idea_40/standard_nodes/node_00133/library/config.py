import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_40"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for the available vCPUs

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    # f_27 is split into 10 characters
    SEQ_LEN = 10
    # 26 uppercase letters + 1 for potential unknown/padding = 27
    VOCAB_SIZE = 27
    # Features f_00 to f_30 excluding f_27 (31 total - 1 categorical = 30 continuous)
    NUM_CONT_FEATURES = 30

    # --------------------------------------------------------------------------
    # Model Architecture: Homogeneous Direct-SwiGLU Hybrid
    # --------------------------------------------------------------------------
    # Stream 1: Categorical Sequence (Transformer)
    EMBED_DIM = 32
    ENCODER_LAYERS = 2
    ENCODER_HEADS = 4
    ENCODER_DROPOUT = 0.1

    # Initialization
    POS_EMBED_STD = 0.02
    EMBED_INIT_STD = 1.0

    # Stream 2 is raw continuous features (no specific params needed here)

    # Fusion & Backbone: LayerNorm SwiGLU ResFunnel
    # Linear Stem maps to first stage width
    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3
    BACKBONE_DROPOUT = 0.35
    STOCHASTIC_DEPTH_MAX = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 15  # Reduced to fit runtime

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY_GROUP1 = 1e-2  # Weights
    WEIGHT_DECAY_GROUP2 = 0.0  # Biases, LayerNorm, PosEmbed

    # Scheduler (Step Decay)
    SCHEDULER_STEP_SIZE = 5  # Decay every 5 epochs
    SCHEDULER_GAMMA = 0.1

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-5

    def __str__(self):
        return str(self.__class__.__dict__)
