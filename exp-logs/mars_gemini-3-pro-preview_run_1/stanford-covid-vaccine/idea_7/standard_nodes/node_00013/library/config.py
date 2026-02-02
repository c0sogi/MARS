import os
import torch


class Config:
    """
    Configuration for the RNA Degradation Prediction Task (Idea 7).
    Implements the settings for the Hybrid Dilated ResNet-BiGRU strategy.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging

    # =========================================================================
    # Paths
    # =========================================================================
    # Input Metadata (Parquet files generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Original Input (for reference/submission format)
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files (for deterministic data processing)
    # Using .npz for efficient numpy array storage
    CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data.npz")
    CACHE_VAL = os.path.join(WORKING_DIR, "val_data.npz")
    CACHE_TEST = os.path.join(WORKING_DIR, "test_data.npz")

    # =========================================================================
    # Data Dimensions & Vocabularies
    # =========================================================================
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Vocabulary Sizes
    # Sequence: A, G, U, C -> 4 (Indices 0-3)
    VOCAB_SIZE_SEQ = 4
    # Structure: (, ), . -> 3 (Indices 0-2)
    VOCAB_SIZE_STRUCT = 3
    # Loop Type: B, E, H, I, M, S, X -> 7 (Indices 0-6)
    VOCAB_SIZE_LOOP = 7

    # =========================================================================
    # Model Architecture: Hybrid Dilated ResNet-BiGRU
    # =========================================================================
    # Embedding Layer
    EMBED_DIM = 128  # Dimension for each input channel (Seq, Struct, Loop)

    # Stage 1: Dilated ResNet Encoder
    # Stacking blocks with exponentially increasing dilation to expand receptive field
    RESNET_BLOCKS = 5
    RESNET_KERNEL_SIZE = 3
    # Dilation rates: [1, 2, 4, 8, 16]
    DILATION_RATES = [2**i for i in range(RESNET_BLOCKS)]
    RESNET_FILTERS = 128
    RESNET_DROPOUT = 0.1

    # Stage 2: Bidirectional GRU
    # Captures global context and sequential dependencies
    GRU_HIDDEN_SIZE = 256
    GRU_LAYERS = 3
    GRU_DROPOUT = 0.3

    # Output Head
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Compute: 1 NVIDIA A100 (40GB) -> Can handle decent batch size
    BATCH_SIZE = 64

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Training Loop
    EPOCHS = 25
    PATIENCE = 7  # Early stopping patience

    # Scheduler: Cosine Annealing
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Loss Function
    # Using MSE (L2) as per Lesson 00012 (Robust losses degrade RMSE metric)
    LOSS_FN = "MSE"

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories.
        Should be called at the start of the execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        # print(f"Directories ensured: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")
