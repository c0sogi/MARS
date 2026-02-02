import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory & File Path Configurations
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_39"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output & Cache Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

    # --------------------------------------------------------------------------
    # General Configurations
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for the available vCPUs

    # --------------------------------------------------------------------------
    # Data Configurations
    # --------------------------------------------------------------------------
    # f_27 decomposition: 10 characters
    SEQUENCE_LENGTH = 10
    # Vocab size for A-Z characters.
    # Indices 0-25 are sufficient, but we allocate a small buffer or use 1-based indexing if needed.
    # Setting to 30 covers A-Z comfortably.
    VOCAB_SIZE = 30

    # 30 Continuous features (f_00 to f_30, excluding f_27)
    NUM_CONTINUOUS_FEATURES = 30

    # --------------------------------------------------------------------------
    # Model Architecture Configurations
    # --------------------------------------------------------------------------
    # Stream 1: Categorical (Post-Norm Transformer)
    EMBED_DIM = 32
    EMBED_INIT_STD = 1.0  # Unit Variance for robust signal propagation
    POS_EMBED_STD = 0.02  # Low Variance Random Noise

    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "gelu"
    TRANSFORMER_NORM_FIRST = False  # Post-Normalization

    # Stream 2: Continuous (Raw Normalized)
    # No specific params other than input dimension defined above

    # Backbone: LayerNorm SwiGLU ResFunnel
    # Stages: 512 -> 256 -> 128
    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3
    BACKBONE_DROPOUT = 0.35
    STOCHASTIC_DEPTH_MAX = 0.2  # Linear schedule 0.0 -> 0.2

    # --------------------------------------------------------------------------
    # Training Configurations
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 40
    LEARNING_RATE = 1e-3

    # Optimizer: AdamW
    # Weight Decay Groups
    WEIGHT_DECAY_GROUP1 = 1e-2  # Linear, Embeddings, Attention projections
    WEIGHT_DECAY_GROUP2 = 0.0  # Biases, Norms, Positional Embeddings

    # Scheduler: Step Decay
    SCHEDULER_STEP_SIZE = 10  # Decay every 10 epochs
    SCHEDULER_GAMMA = 0.1  # Decay factor
