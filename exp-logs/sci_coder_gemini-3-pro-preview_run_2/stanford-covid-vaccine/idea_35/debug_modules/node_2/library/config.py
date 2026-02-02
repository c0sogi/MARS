import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for caching processed features and model checkpoints
WORKING_DIR = "./working/idea_35"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Submission Paths
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATASET CONSTANTS
# =============================================================================
SEQ_LENGTH = 107
SEQ_SCORED = 68

# Target columns in the order provided in the dataset/submission
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

# The subset of columns that are actually scored in the competition metric
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

# Indices of the scored targets within the TARGET_COLS list
# reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
SCORED_INDICES = [0, 1, 3]

# =============================================================================
# FEATURE DIMENSIONS
# =============================================================================
# Input Channel Calculation:
# 1. Sequence (A, G, C, U) -> 4
# 2. Structure ((, ), .) -> 3
# 3. Predicted Loop Type (S, M, I, B, H, E, X) -> 7
# 4. Partner Base Identity (A, G, C, U) -> 4
INPUT_CHANNELS = 4 + 3 + 7 + 4  # Total: 18

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
HIDDEN_DIM = 64  # Channel dimension for TCN backbone
LATENT_DIM = 64  # Dimension for projected features before fusion
FEEDBACK_DIM = 32  # Dimension for feedback embedding
DROPOUT = 0.1  # Dropout rate for backbone and feedback
DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation rates for TCN
KERNEL_SIZE = 3  # Kernel size for convolutions

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
SEED = 42
NUM_WORKERS = 2  # Number of dataloader workers

# Loss Weights for the feedback mechanism
LOSS_WEIGHT_PASS_2 = 1.0  # Weight for the final refined prediction
LOSS_WEIGHT_PASS_1 = 0.5  # Weight for the initial prediction (auxiliary loss)


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================
class Config:
    """
    Configuration class to encapsulate hyperparameters and allow for
    overrides (e.g., for debugging or hyperparameter tuning).
    """

    def __init__(self, debug=False, **kwargs):
        # Paths
        self.input_dir = INPUT_DIR
        self.metadata_dir = METADATA_DIR
        self.working_dir = WORKING_DIR
        self.train_csv = TRAIN_CSV
        self.val_csv = VAL_CSV
        self.test_csv = TEST_CSV
        self.submission_path = SUBMISSION_PATH

        # Data
        self.seq_length = SEQ_LENGTH
        self.seq_scored = SEQ_SCORED
        self.target_cols = TARGET_COLS
        self.scored_targets = SCORED_TARGETS
        self.scored_indices = SCORED_INDICES
        self.input_channels = INPUT_CHANNELS

        # Model
        self.hidden_dim = HIDDEN_DIM
        self.latent_dim = LATENT_DIM
        self.feedback_dim = FEEDBACK_DIM
        self.dropout = DROPOUT
        self.dilations = DILATIONS
        self.kernel_size = KERNEL_SIZE

        # Training
        self.batch_size = BATCH_SIZE
        self.lr = LEARNING_RATE
        self.epochs = EPOCHS
        self.seed = SEED
        self.num_workers = NUM_WORKERS
        self.loss_weight_pass_2 = LOSS_WEIGHT_PASS_2
        self.loss_weight_pass_1 = LOSS_WEIGHT_PASS_1

        # Debug Mode Overrides
        if debug:
            self.epochs = 2
            self.batch_size = 8
            # In debug mode, we reduce epochs and batch size for quick validation

        # Allow arbitrary overrides via kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)
