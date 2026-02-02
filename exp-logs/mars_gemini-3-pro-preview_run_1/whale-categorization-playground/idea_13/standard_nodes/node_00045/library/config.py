import os
import torch


class Config:
    """
    Global configuration for the Whale Identification pipeline.
    Implements the settings for the Heterogeneous Ensemble with Iterative Self-Training.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"

    # Sub-directories (created automatically via setup)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Data Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 320  # Fixed resolution: 320x320
    NUM_CLASSES = 4029  # Total unique classes in training set
    NUM_WORKERS = 4  # Number of DataLoader workers
    PIN_MEMORY = True

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Heterogeneous Ensemble Backbones
    MODEL_NAMES = ["densenet121", "resnet50_ibn_a"]

    EMBEDDING_SIZE = 512  # Dimension of the embedding vector before ArcFace
    DROPOUT = 0.0  # Explicitly excluded to prevent underfitting

    # ArcFace Head Parameters
    ARCFACE_S = 30.0  # Scale factor
    ARCFACE_M = 0.5  # Margin

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32  # Fits 320x320 on A100 easily (could go higher, but 32 is safe)
    EPOCHS = 20  # Sufficient for convergence with pre-trained models
    LEARNING_RATE = 3e-4  # Conservative LR for AdamW
    WEIGHT_DECAY = 1e-4  # Regularization for AdamW
    LABEL_SMOOTHING = 0.1  # Critical for singleton classes
    PATIENCE = 5  # Early stopping patience

    # -------------------------------------------------------------------------
    # Self-Training / Pseudo-Labeling
    # -------------------------------------------------------------------------
    PSEUDO_LABEL_THRESHOLD = 0.9  # High confidence threshold

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures all working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_batch_size(cls, inference=False):
        """
        Returns batch size, optionally larger for inference.
        """
        if inference:
            return cls.BATCH_SIZE * 2
        return cls.BATCH_SIZE


# Automatically setup directories when config is imported
Config.setup()
