import os
import torch


class Config:
    """
    Configuration module for the RNA Degradation Prediction task.
    Implements settings for the Dense-Context Latent-Refined Hybrid Network (Idea 9).
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Input Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (NPZ)
    # Distinct filenames used to ensure cache invalidation/safety for Idea 9
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_dense_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_dense_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_dense_v1.npz")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SCORED_SEQ_LENGTH = 68

    # Target Columns
    # All 5 columns are required for submission
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these 3 are used for the MCRMSE loss calculation
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = len(TARGET_COLS)

    # Input Feature Dimensions
    # 1. Sequence One-Hot: 4 (A, G, C, U)
    # 2. Structure One-Hot: 3 (., (, ))
    # 3. Loop Type One-Hot: 7 (S, M, I, B, H, E, X)
    # 4. Partner Base One-Hot: 5 (A, G, C, U, None)
    INPUT_CHANNELS = 4 + 3 + 7 + 5  # 19

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # DenseNet Backbone (Dense Dilated TCN)
    GROWTH_RATE = 32
    KERNEL_SIZE = 3
    # Exponential dilation schedule for global receptive field
    DILATION_SCHEDULE = [1, 2, 4, 8, 16, 32]

    # Latent Refinement
    # Dimension to compress dense features to before the Latent Gather step
    BOTTLENECK_DIM = 64

    # Recurrent Head (BiGRU)
    # Input to RNN is concatenation of Local (Bottleneck) + Gathered (Bottleneck)
    RNN_INPUT_DIM = BOTTLENECK_DIM * 2  # 128
    # Constraint: Hidden size strictly set to input_dim // 2 to prevent overfitting
    RNN_HIDDEN_DIM = RNN_INPUT_DIM // 2  # 64
    RNN_LAYERS = 1
    RNN_DROPOUT = 0.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience

    # Hardware
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, debug=False, batch_size=None, epochs=None):
        """
        Initialize configuration with optional overrides.

        Args:
            debug (bool): If True, enables debug mode with reduced epochs and batch size.
            batch_size (int, optional): Override the default batch size.
            epochs (int, optional): Override the default number of epochs.
        """
        self.debug = debug

        if batch_size is not None:
            self.BATCH_SIZE = batch_size

        if epochs is not None:
            self.EPOCHS = epochs

        if self.debug:
            self.EPOCHS = 2
            self.BATCH_SIZE = 4
            # Debug mode implies faster iteration, potentially on smaller data subsets
            # handled by the dataset loader.

    @classmethod
    def setup(cls):
        """
        Prepare the working directory for cache and outputs.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
