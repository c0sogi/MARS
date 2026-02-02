import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Adjust number of workers based on available vCPUs (12)
    NUM_WORKERS = 8

    # --------------------------------------------------------------------------
    # Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_37"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Stratified Splits)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    TARGET_COL = "target"
    ID_COL = "id"

    # Feature Definitions
    CAT_FEATURE = "f_27"
    # f_27 is split into 10 characters.
    # Vocabulary: A-Z (26) + Padding/Unknown. Safe size = 30.
    CAT_VOCAB_SIZE = 30
    CAT_SEQ_LEN = 10

    # Continuous features: f_00 to f_30, excluding f_27
    CONT_FEATURES = [f"f_{i:02d}" for i in range(31) if i != 27]
    NUM_CONT_FEATURES = len(CONT_FEATURES)

    # --------------------------------------------------------------------------
    # Model Architecture: Interface-Normalized Hybrid SwiGLU Network
    # --------------------------------------------------------------------------
    # Stream 1: Categorical (Stabilized Post-Norm Transformer)
    EMBED_DIM = 32
    EMBED_INIT_STD = 1.0  # Unit Variance Initialization
    POS_EMBED_INIT_STD = 0.02  # Low Variance Random Noise

    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "gelu"
    TRANSFORMER_NORM_FIRST = False  # Post-Normalization

    # Stream 2: Continuous
    # Features are Z-scored and fused directly via Linear Stem

    # Backbone: SwiGLU ResFunnel
    # Stages: 512 -> 256 -> 128
    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3

    # Regularization
    STOCHASTIC_DEPTH_MAX = 0.2  # Linear schedule 0.0 -> 0.2
    MAIN_DROPOUT = 0.35

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 1024
    EPOCHS = 40
    LEARNING_RATE = 1e-3

    # Optimizer (Decoupled Weight Decay)
    WEIGHT_DECAY_PARAMS = 1e-2  # For Linear, Embeddings, Attention
    WEIGHT_DECAY_BIAS_NORM = 0.0  # For Biases, LayerNorm, PosEmbed

    # Scheduler (Aggressive Step Decay)
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # --------------------------------------------------------------------------
    # Runtime Control
    # --------------------------------------------------------------------------
    # Set to an integer (e.g., 10000) to limit training data for debugging
    # Set to None to use the full dataset
    DEBUG_SAMPLE_SIZE = None
