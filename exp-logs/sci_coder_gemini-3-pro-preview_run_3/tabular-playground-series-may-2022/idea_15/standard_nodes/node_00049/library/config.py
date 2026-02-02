import os
import torch


class Config:
    """
    Central configuration for the Input-Attentive Swish Funnel Network experiment.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for Idea 15
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Metadata Input Files (Stratified Splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files (Parquet/Numpy for fast loading)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")
    METADATA_CACHE = os.path.join(
        WORKING_DIR, "metadata.npy"
    )  # For vocab sizes, scaler params, etc.

    # ==========================================
    # Data Preprocessing
    # ==========================================
    # Feature Engineering Config
    N_CHAR_POSITIONS = 10  # f_27 decomposed length

    # ==========================================
    # Model Architecture
    # ==========================================
    # Input Embeddings
    EMBEDDING_DIM = 16

    # Input Attention Block
    ATTN_BOTTLENECK_DIM = 64

    # Swish Funnel Backbone
    HIDDEN_LAYERS = [512, 256, 128]
    DROPOUT_RATE = 0.2

    # Output
    OUTPUT_DIM = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    MAX_LR = 1e-2  # For OneCycleLR
    WEIGHT_DECAY = 1e-5  # Calibrated for tabular
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # Hardware
    # ==========================================
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
