import os
import torch


class Config:
    # =========================================================================
    # System Configuration
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Input Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    # Using 'idea_5' as the designated working folder for this run
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "stoichiometry_encoder.pth")
    TRAIN_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "train_embeddings.npy")
    TRAIN_LABELS_CACHE_PATH = os.path.join(WORKING_DIR, "train_labels_cache.npy")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Chemical Domain Constants
    # =========================================================================
    # Atoms to track for the stoichiometry regression task
    ATOM_LIST = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "I"]
    NUM_ATOMS = len(ATOM_LIST)

    # =========================================================================
    # Data / Image Hyperparameters
    # =========================================================================
    IMAGE_SIZE = 256  # Input resolution (256x256)
    IN_CHANNELS = 3  # Using RGB (could be 1 for grayscale, but backbones expect 3)
    NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    NORM_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    BACKBONE = "efficientnet_b0"  # Lightweight, efficient backbone
    EMBEDDING_DIM = 256  # Dimension of the retrieval vector
    PRETRAINED = True  # Use ImageNet pretrained weights

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 128  # Batch size for training
    VAL_BATCH_SIZE = 256  # Batch size for validation/inference
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-5  # L2 regularization
    EPOCHS = 10  # Number of training epochs
    PATIENCE = 3  # Early stopping patience

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SAMPLE_SIZE = 5000

    @staticmethod
    def setup_directories():
        """Ensures necessary output directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup_directories()
