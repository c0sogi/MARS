import os
import torch


class Config:
    """
    Global configuration for the Bird Species Classification Task.
    Implements settings for a Hybrid Neuro-Symbolic Ensemble (CNNs + MLP).
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of data loading workers

    # ==========================================
    # Task Specifics
    # ==========================================
    NUM_CLASSES = 19
    NUM_FOLDS = 5

    # ==========================================
    # Paths
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Data Sources
    # Stream A: Standard Spectrograms
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
    # Stream B: Bag-of-Audio-Words Features
    HISTOGRAM_FILE = os.path.join(
        INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )

    # Output Directories (Write Allowed)
    WORKING_DIR = "./working/idea_26"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Preprocessing & Augmentation
    # ==========================================
    # Image Settings
    IMAGE_SIZE = (224, 224)
    INPUT_CHANNELS = 3  # Replicate single channel to 3 for pre-trained models

    # Augmentation Hyperparameters
    MIXUP_ALPHA = 0.4
    SHIFT_LIMIT = 0.1  # Max horizontal shift (<10% of width)
    BRIGHTNESS_LIMIT = 0.2  # Photometric augmentation
    CONTRAST_LIMIT = 0.2  # Photometric augmentation

    # ==========================================
    # Model Architectures
    # ==========================================
    # Deep Learning Stream (CNNs)
    CNN_ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]

    # Shallow Learning Stream (MLP)
    MLP_INPUT_DIM = 100  # Based on k=100 clusters in histogram features
    MLP_HIDDEN_DIM = 128
    MLP_DROPOUT = 0.5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    MAX_EPOCHS = 100  # High ceiling, controlled by Early Stopping
    PATIENCE = 15  # Aggressive early stopping

    # Optimization - CNNs
    LR_CNN = 1e-3
    WEIGHT_DECAY_CNN = 1e-2  # AdamW weight decay

    # Optimization - MLP
    LR_MLP = 1e-3
    WEIGHT_DECAY_MLP = 0.0  # Standard Adam

    # Ensemble Strategy
    TOP_K_CHECKPOINTS = 3  # Snapshot averaging per fold

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories for caching, checkpoints, and submissions.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR} and {cls.SUBMISSION_DIR}")
