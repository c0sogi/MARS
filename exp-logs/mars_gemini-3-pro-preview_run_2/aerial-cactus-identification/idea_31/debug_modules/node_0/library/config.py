import os


class Config:
    """
    Configuration module for the Cactus Classification Task (Idea 31).
    Implements parameters for the Custom Wide SE-Res2NeXt with Multi-Scale Aggregation.
    """

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_31"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility & Ensembling
    # ==========================================
    # Homogeneous Seed Averaging: 5 independent seeds
    SEEDS = [0, 1, 2, 3, 4]

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 32
    NUM_CLASSES = 1
    NUM_WORKERS = 4  # Number of DataLoader workers

    # Debugging / Development
    # Set to an integer (e.g., 1000) to limit dataset size for rapid testing
    # Set to None for full training
    MAX_TRAIN_SAMPLES = None
    DEBUG = False

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 20
    BATCH_SIZE = 128  # A100 allows larger batches; 128 is stable for Wide architectures

    # Optimizer: AdamW
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler: Cosine Annealing
    # T_max corresponds to total epochs
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Model Architecture: Custom Wide SE-Res2NeXt
    # ==========================================
    MODEL_CONFIG = {
        # "Super-Wide" Channel Configuration: [64, 128, 256]
        # Stage 1: 32x32 output
        # Stage 2: 16x16 output
        # Stage 3: 8x8 output (Strictly preserved)
        "stage_channels": [64, 128, 256],
        # ResNeXt: Grouped Convolutions
        "cardinality": 32,
        # Res2Net: Hierarchical Multi-Scale Connections
        "res2net_scale": 4,
        # Squeeze-and-Excitation: Reduction Ratio
        "se_reduction": 16,
        # Multi-Scale Aggregation Head
        # Concatenate Global Average Pooled features from:
        # Index 1 (Stage 2, 16x16) and Index 2 (Stage 3, 8x8)
        "aggregation_stages": [1, 2],
        "num_classes": 1,
    }

    # ==========================================
    # Inference & TTA
    # ==========================================
    # Test Time Augmentation: Original, Horizontal Flip, Vertical Flip
    TTA_ENABLED = True

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working directories exist.
        This is called automatically upon module import.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup to create directories immediately
Config.setup()
