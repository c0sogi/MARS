import os
import torch


class Config:
    """
    Global configuration for the Structurally Diverse Parallel Ensemble (SDPE) pipeline.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific directory for this experiment (Idea 20)
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_20")

    # Source Data (Metadata Splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Original Data (For Transductive Vocabulary Alignment)
    ORIGINAL_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    ORIGINAL_TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Artifacts
    MODEL_PATH = os.path.join(IDEA_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(IDEA_DIR, "submission.csv")

    # Caching Paths (Parquet for dataframes, NPY for metadata)
    CACHE_TRAIN = os.path.join(IDEA_DIR, "train_processed.parquet")
    CACHE_VAL = os.path.join(IDEA_DIR, "val_processed.parquet")
    CACHE_TEST = os.path.join(IDEA_DIR, "test_processed.parquet")
    CACHE_METADATA = os.path.join(IDEA_DIR, "metadata.npy")

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 1024
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3  # Max LR for OneCycle Policy
    WEIGHT_DECAY = 1e-5  # Explicitly set for Adam (Standard)

    # -------------------------------------------------------------------------
    # Model Architecture (SDPE)
    # -------------------------------------------------------------------------
    EMBED_DIM = 16

    # Stream Configurations
    # 1 & 2: Anchor (Standard ReLU MLP)
    # 3: Smooth Manifold (Wide + SiLU)
    # 4: Feature Selection (GLU)
    # 5: Deep & Robust (Deep ReLU MLP)
    STREAM_CONFIGS = [
        {
            "name": "anchor_1",
            "type": "mlp",
            "layers": [512, 256, 128],
            "act": "ReLU",
            "dropout": 0.20,
        },
        {
            "name": "anchor_2",
            "type": "mlp",
            "layers": [512, 256, 128],
            "act": "ReLU",
            "dropout": 0.20,
        },
        {
            "name": "smooth",
            "type": "mlp",
            "layers": [1024, 512, 256],
            "act": "SiLU",
            "dropout": 0.20,
        },
        {
            "name": "glu_net",
            "type": "glu",
            "layers": [512, 256, 128],
            "act": "None",
            "dropout": 0.30,
        },
        {
            "name": "deep",
            "type": "mlp",
            "layers": [512, 512, 256, 128],
            "act": "ReLU",
            "dropout": 0.25,
        },
    ]

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
