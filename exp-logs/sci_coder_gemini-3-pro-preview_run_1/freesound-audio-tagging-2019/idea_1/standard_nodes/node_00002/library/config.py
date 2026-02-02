import os
import torch


class Config:
    """
    Configuration class for the Audio Tagging task.
    Centralizes hyperparameters for data processing, model training, and file paths.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    SR = 32000  # Sample Rate (32 kHz)
    DURATION = 20  # Duration of audio clips in seconds. Increased to cover majority of signal. Cite {solution_lesson_node_00001}
    N_MELS = 64  # Number of Mel bands
    N_FFT = 1024  # FFT window size (~32ms)
    HOP_LENGTH = 320  # Hop length (~10ms)
    FMIN = 20  # Minimum frequency
    FMAX = 16000  # Maximum frequency (Nyquist)

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    NUM_CLASSES = 80  # Number of target categories

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64  # Batch size for training
    LEARNING_RATE = 1e-3  # Initial learning rate for Adam optimizer
    MAX_EPOCHS = 30  # Maximum number of training epochs
    PATIENCE = 5  # Early stopping patience

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SUBSET_SIZE = 200  # Number of samples to use when DEBUG is True

    # ==========================================
    # Compute Environment
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8  # Number of DataLoader workers (12 vCPUs available)

    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Model Checkpoint and Submission Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
