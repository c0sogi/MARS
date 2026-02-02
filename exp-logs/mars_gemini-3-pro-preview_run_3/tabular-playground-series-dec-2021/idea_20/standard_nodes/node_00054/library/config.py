import os


class Config:
    """
    Configuration for the Parallel Vector-DCN-ResNet experiment.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"

    # Data Paths (using metadata parquets as requested)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --------------------------------------------------------------------------
    # Data Definitions
    # --------------------------------------------------------------------------
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Raw Continuous Features
    CONTINUOUS_COLS = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    # Raw Binary Features (Wilderness Areas and Soil Types)
    # Generated dynamically to match dataset columns
    WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
    SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]
    BINARY_COLS = WILDERNESS_COLS + SOIL_COLS

    # --------------------------------------------------------------------------
    # Feature Engineering Flags
    # --------------------------------------------------------------------------
    # Cyclical Augmentation: Calculate Aspect_Sin, Aspect_Cos but RETAIN raw Aspect
    USE_CYCLICAL_ASPECT = True

    # Geometric Magnitude: Sqrt(H_Dist^2 + V_Dist^2) for Hydrology
    USE_EUCLIDEAN_HYDRO = True

    # Directional Preservation: Elevation - Vertical_Distance_To_Hydrology
    USE_ABS_HYDRO_ELEV = True

    # Global Context: Mean of distances to Hydrology, Roadways, Fire Points
    USE_MEAN_AMENITIES = True

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "ParallelVectorDCNResNet"

    # Branch 1: Vector-Based Cross Layer
    CROSS_LAYER_TYPE = "Vector"  # Rank-1 interaction: x_0 * (x_l^T w) + b + x_l

    # Branch 2: Wide ResNet Backbone
    HIDDEN_DIM = 512  # Width of the ResNet layers
    RESNET_BLOCKS = 2  # Number of residual blocks
    ACTIVATION = "ReLU"

    # General
    DROPOUT_RATE = 0.1
    NUM_CLASSES = 7  # Cover_Type 1-7

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3

    # Optimization
    OPTIMIZER = "AdamW"
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "CosineAnnealing"  # Decays to 0 over EPOCHS

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Ensures necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
