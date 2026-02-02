import os
import torch


class Config:
    """
    Configuration for the FiLM-Conditioned Dual-Axis Network pipeline.
    """

    # ==========================================
    # 1. General Setup
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # 2. Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Processing
    # ==========================================
    IMG_SIZE = 224
    NUM_SLABS = 3
    SLAB_OVERLAP = 0.15

    # Tabular Data Configuration
    # Features: Age (1) + Percent (1) + Sex (2, OHE) + SmokingStatus (3, OHE)
    TABULAR_COLS = ["Age", "Percent", "Sex", "SmokingStatus"]
    TABULAR_INPUT_DIM = 7

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    # EfficientNet-B0 final feature map channels (before pooling)
    BACKBONE_OUT_CHANNELS = 1280

    # FiLM (Feature-wise Linear Modulation) settings
    FILM_HIDDEN_DIM = 256

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 8

    # ==========================================
    # 6. Metric & Inference
    # ==========================================
    # Metric constraints
    MAX_ERROR_CLIP = 1000.0
    MIN_CONFIDENCE_CLIP = 70.0

    # ==========================================
    # 7. Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print("PIPELINE CONFIGURATION")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<25}: {v}")
        print("=" * 40 + "\n")
