import os
import torch


class Config:
    # ==========================================
    # Experiment Control
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to train on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 10000

    # ==========================================
    # Hardware & Compute
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available CPUs, capping at a reasonable number for data loading
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    # Metadata directories (Pre-split Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working directory for caching and checkpoints
    WORKING_DIR = "./working/idea_12"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Raw Continuous Features (Input to normalization and augmentation)
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

    # Raw Binary Features (Input to DCN and Concatenation)
    # Wilderness Areas 1-4 and Soil Types 1-40
    BINARY_COLS = [
        "Wilderness_Area1",
        "Wilderness_Area2",
        "Wilderness_Area3",
        "Wilderness_Area4",
    ] + [f"Soil_Type{i}" for i in range(1, 41)]

    # Names of features generated during engineering (for tracking dimensions)
    ENGINEERED_COLS = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Distance_To_Hydrology",
        "Absolute_Hydrology_Elevation",
        "Mean_Distance_To_Amenities",
    ]

    # ==========================================
    # Model Architecture: Parallel DCN-ResNeXt
    # ==========================================
    # Branch 1: Vector-DCN
    DCN_LAYERS = 3

    # Branch 2: ResNeXt Backbone
    RESNEXT_HIDDEN_DIM = 1024  # Scaled width
    RESNEXT_GROUPS = 32  # Cardinality
    RESNEXT_LAYERS = 2  # Depth of residual blocks

    # General
    DROPOUT = 0.1
    NUM_CLASSES = 7  # Handling classes 1-7 (mapped to 0-6 internally)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler: Cosine Annealing
    ETA_MIN = 0.0

    # Optimization
    PATIENCE = 10  # Early stopping patience
