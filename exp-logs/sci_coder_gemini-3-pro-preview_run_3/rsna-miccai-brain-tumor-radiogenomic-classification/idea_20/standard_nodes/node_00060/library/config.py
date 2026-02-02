import os


class Config:
    """
    Configuration for the Modality-Grouped Stabilized High-Density (MG-SHD) Network.
    """

    # ==========================================
    # Data Specifications
    # ==========================================
    # Number of slices to sample per modality (Uniform Sampling)
    NUM_SLICES_PER_MODALITY = 32

    # Number of MRI modalities (FLAIR, T1w, T1wCE, T2w)
    NUM_MODALITIES = 4

    # Input image size (256x256)
    IMG_SIZE = 256

    # ==========================================
    # Model Architecture
    # ==========================================
    # Total input channels: 32 slices * 4 modalities = 128
    IN_CHANS = NUM_SLICES_PER_MODALITY * NUM_MODALITIES

    # Output channels of the Stabilized Compression Stem
    STEM_CHANS = 64

    # Backbone architecture (timm)
    BACKBONE = "efficientnet_b0"

    # Drop Path Rate for Stochastic Depth
    DROP_PATH_RATE = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch size (A100 40GB can handle 32 with this input size)
    BATCH_SIZE = 32

    # Learning Rate (Adam)
    LEARNING_RATE = 1e-4

    # Number of training epochs
    EPOCHS = 15

    # Random Seed for reproducibility
    SEED = 42

    # ==========================================
    # Paths & Directories
    # ==========================================
    # Root Input Directory
    INPUT_DIR = "./input"

    # Metadata Directory and Files
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_20"

    # Cache Directory for processed arrays
    CACHE_DIR = WORKING_DIR

    # Path to save the best model checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
