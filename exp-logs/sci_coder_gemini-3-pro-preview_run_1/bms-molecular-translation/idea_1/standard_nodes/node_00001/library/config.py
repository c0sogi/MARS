import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the InChI prediction project.
    Defines hyperparameters, file paths, and reproducibility settings.
    """

    # ==========================================
    # Reproducibility & Compute
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # ==========================================
    # Workflow Control
    # ==========================================
    DEBUG = False  # Set to True to train on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 2000  # Number of samples to use in debug mode

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    TOKENIZER_PATH = os.path.join(
        WORKING_DIR, "tokenizer.npy"
    )  # Using npy for caching vocab
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = 256  # Input image resolution (256x256)
    IN_CHANNELS = 3  # ResNet backbone expects 3 channels (RGB)

    # Sequence Parameters
    # Max length observed in EDA was 403.
    # We add margin for special tokens (<sos>, <eos>) and potential padding.
    MAX_LEN = 410

    # Vocabulary Special Tokens
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    # ==========================================
    # Model Architecture (Show and Tell)
    # ==========================================
    # Encoder: CNN Backbone
    ENCODER_MODEL = "resnet18"
    ENCODER_DIM = 512  # Feature dimension from ResNet18 global pooling

    # Decoder: RNN (GRU)
    EMBED_DIM = 256  # Dimension of character embeddings
    HIDDEN_SIZE = 512  # Hidden state size of the GRU
    DROPOUT = 0.5  # Dropout probability

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Optimization & Scheduling
    CLIP_GRAD = 5.0  # Gradient clipping max norm
    PATIENCE = 5  # Early stopping patience (epochs)
    TEACHER_FORCING_RATIO = (
        0.5  # Probability of using ground truth as input during training
    )

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets fixed random seeds for Python, NumPy, and PyTorch to ensure
        reproducible results across runs.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms can be slower, but ensure reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize reproducibility settings immediately upon import
Config.setup_reproducibility(Config.SEED)
