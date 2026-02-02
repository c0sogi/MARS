import os
import torch


class Config:
    """
    Configuration class for the Plant Species Classification task.
    Centralizes all parameters for paths, model architecture, and training strategy.
    """

    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Generated Metadata Paths (ReadOnly)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output/Cache Paths (Writeable)
    # Stores the mapping from category_id (species) to genus_id for taxonomic loss
    TAXONOMY_MAP_PATH = os.path.join(WORKING_DIR, "taxonomy_mapping.parquet")
    # Stores model checkpoints during training
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    # Path for the final submission file
    SUBMISSION_PATH = "./submission/submission.csv"

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b3"
    # Total number of species classes in the dataset
    NUM_CLASSES = 32093

    # ArcFace Head Parameters
    EMBEDDING_SIZE = 512
    ARCFACE_SCALE = 30.0
    ARCFACE_MARGIN = 0.50

    # Generalized Mean Pooling (GeM) Parameter (initial p value)
    GEM_P = 3.0

    # Dropout rate for the backbone features
    DROPOUT_RATE = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilize available vCPUs for data loading
    NUM_WORKERS = 12

    # Loss Configuration
    FOCAL_LOSS_GAMMA = 2.0
    # Probability mass distributed among sibling species in the same genus
    LABEL_SMOOTHING_EPS = 0.1

    # Progressive Resizing Strategy
    # Phase 1: Faster training with smaller images to learn coarse features
    PHASE1 = {
        "name": "phase1",
        "img_size": 224,
        "batch_size": 128,  # A100 40GB allows large batch size for B3 @ 224
        "epochs": 6,  # Sufficient epochs for coarse convergence
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler_patience": 1,
        "scheduler_factor": 0.5,
    }

    # Phase 2: Fine-tuning with larger images for fine-grained details
    PHASE2 = {
        "name": "phase2",
        "img_size": 300,  # Native resolution for EfficientNet-B3
        "batch_size": 64,  # Reduced batch size for larger resolution
        "epochs": 6,  # Fine-tuning epochs
        "lr": 1e-4,  # Lower learning rate for fine-tuning
        "weight_decay": 1e-4,
        "scheduler_patience": 1,
        "scheduler_factor": 0.5,
    }

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set to True to train on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 5000

    @classmethod
    def create_directories(cls):
        """Creates necessary working directories if they do not exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)


# Automatically create directories when config is imported
Config.create_directories()
