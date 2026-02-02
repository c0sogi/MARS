import os
import torch


class Config:
    """
    Central configuration for the Phase-Aware Stratified 1D Attention ResUNet with ASPP.
    """

    # --- General Configuration ---
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching Paths (Parquet format)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_SAVE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Preprocessing Hyperparameters ---
    # Stratified Aggregation:
    # 3 Strata (Global, High-Precision, High-Risk)
    # 4 Statistics per feature (Mean, Std, Min, Max)
    # 2 Base Features (Cn0DbHz, SvElevationDegrees)
    # 3 * 4 * 2 = 24 features
    # + 2 Context Features (Azimuth Sin, Azimuth Cos)
    # Total Input Channels = 26
    IN_CHANNELS = 26

    # Targets: Delta North (Meters), Delta East (Meters)
    OUT_CHANNELS = 2

    # Feature Scaling (Approximate max values for normalization)
    CN0_SCALE = 50.0
    ELEV_SCALE = 90.0

    # --- Model Architecture ---
    MODEL_NAME = "ResUNet1D_Attention_ASPP"
    BASE_FILTERS = 64
    DEPTH = 5
    USE_ATTENTION = True  # Enable Attention Gates
    USE_ASPP = True  # Enable Atrous Spatial Pyramid Pooling
    ASPP_DILATIONS = [1, 6, 12, 18]

    # --- Training Hyperparameters ---
    BATCH_SIZE = 4  # Low batch size for sequence data
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    GRAD_CLIP = 1.0  # Gradient clipping max norm

    # Scheduler Params (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # --- Debugging ---
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configuration loaded. Working directory: {cls.WORKING_DIR}")


# Automatically setup directories when config is imported
Config.setup()
