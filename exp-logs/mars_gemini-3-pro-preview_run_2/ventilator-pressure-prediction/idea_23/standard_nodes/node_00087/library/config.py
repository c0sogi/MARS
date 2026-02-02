import os
import torch


class Config:
    """
    Configuration for WPA-BiLSTM (Wide-Bandwidth Physics-Augmented BiLSTM)
    and the Segregated Physics-Fidelity Pipeline.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"

    # Raw Data
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-split)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # 2. Data Pipeline & Feature Engineering
    # ==========================================
    # Continuous features: Apply RobustScaler (Median/IQR)
    # Includes raw inputs, physics interactions, and dynamic deltas
    CONTINUOUS_FEATURES = [
        "time_step",
        "u_in",
        "R",
        "C",
        "area",  # Volume: cumsum(u_in * dt)
        "R__u_in",  # Interaction: R * u_in
        "u_in_cumsum_div_C",  # Interaction: Volume / C
        "u_in_lag1",  # Delta: u_in_t - u_in_{t-1}
        "u_in_lag2",  # Delta: u_in_t - u_in_{t-2}
        "u_in_lag3",  # Delta: u_in_t - u_in_{t-3}
        "u_in_lag4",  # Delta: u_in_t - u_in_{t-4}
    ]

    # Binary features: Keep Raw (No Scaling)
    # Scaling binary features distorts their semantic meaning (0/1)
    BINARY_FEATURES = ["u_out"]

    # Target Variable
    TARGET_COL = "pressure"

    # Derived Dimensions
    INPUT_DIM = len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES)

    # ==========================================
    # 3. Model Architecture (WPA-BiLSTM)
    # ==========================================
    # Backbone
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 4
    BIDIRECTIONAL = True

    # Injection Path
    # Constrained GLU with bottleneck projection (Cite {solution_lesson_node_00045})
    GLU_HIDDEN_SIZE = 128

    # Regularization
    DROPOUT = 0.1

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    # Stretched Horizon Convergence Protocol
    EPOCHS = 200

    # Batch Size (Optimized for A100 40GB)
    BATCH_SIZE = 256

    # Optimizer: AdamW
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler: Cosine Annealing
    T_MAX = 200  # Matches EPOCHS for full stretch
    ETA_MIN = 1e-5  # Minimum LR

    # Loss Function: Weighted L1
    W_INSPIRATORY = 1.0
    W_EXPIRATORY = 0.1

    # ==========================================
    # 5. System & Reproducibility
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize necessary directories for caching and outputs.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic flags for PyTorch
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
