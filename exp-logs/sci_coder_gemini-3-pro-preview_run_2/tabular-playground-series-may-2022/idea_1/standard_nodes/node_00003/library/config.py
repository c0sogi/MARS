import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Cache directory specifically for Idea 1 (Shallow Embedding MLP)
    # Used to store processed tensors/parquets
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    TARGET_COL = "target"
    ID_COL = "id"

    # Feature Definitions
    # Continuous features: f_00 to f_30, excluding f_27
    CONTINUOUS_FEATURES = [f"f_{i:02d}" for i in range(31) if i != 27]
    CATEGORICAL_FEATURE = "f_27"

    # Specifics for f_27 string processing
    SEQUENCE_LENGTH = 10  # The string length of f_27 is fixed at 10
    VOCAB_SIZE = 26  # Characters 'A'-'Z' map to 0-25

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    EMBEDDING_DIM = (
        32  # Dimension size for character embeddings (Cite solution_lesson_node_00002)
    )
    HIDDEN_LAYERS = [
        512,
        256,
        128,
    ]  # Deep architecture (Cite solution_lesson_node_00002)
    OUTPUT_DIM = 1  # Binary output
    DROPOUT = 0.1  # Dropout probability

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW optimizer
    EPOCHS = 30  # Maximum number of training epochs
    EARLY_STOPPING_PATIENCE = 5  # Stop if validation AUC doesn't improve

    # --------------------------------------------------------------------------
    # Hardware & Runtime Settings
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, 4 workers is usually a safe sweet spot for DataLoader
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Debugging / Development Flags
    # --------------------------------------------------------------------------
    DEBUG = False  # Set to True to train on a small subset
    DEBUG_SUBSET_SIZE = 10000  # Number of samples for debugging

    @classmethod
    def setup(cls):
        """
        Initialize the necessary directories for the project.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
