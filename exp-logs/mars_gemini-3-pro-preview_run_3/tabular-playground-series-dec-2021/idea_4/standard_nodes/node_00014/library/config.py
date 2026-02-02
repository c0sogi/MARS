import os
import torch


class Config:
    """
    Global configuration for the Forest Cover Type prediction pipeline.
    """

    # ==========================================
    # 1. Random Seed & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    # Input Metadata (Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Directories
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 3. Data Configuration
    # ==========================================
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Feature Definitions
    # Continuous features to be standardized
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

    # Features used for composite feature engineering (Sum/Mean)
    DISTANCE_COLS = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]

    # Binary features (One-hot encoded in original data)
    WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
    SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]
    BINARY_COLS = WILDERNESS_COLS + SOIL_COLS

    # Total input dimension will be calculated dynamically, but roughly:
    # 10 continuous + 4 wilderness + 40 soil + engineered features

    # Number of target classes (Cover_Type 1-7)
    # We will map these to 0-based indices internally if necessary
    NUM_CLASSES = 7

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    # Denoising Autoencoder (DAE)
    LATENT_DIM = 64  # Dimension of the encoded representation
    SWAP_NOISE_PROB = 0.15  # Probability of swapping a feature value

    # ResNet-MLP Classifier
    HIDDEN_DIM = 256  # Width of hidden layers
    DROPOUT = 0.3  # Dropout rate

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024

    # Stage 1: Unsupervised Pretraining (DAE)
    LR_PRETRAIN = 1e-3
    EPOCHS_PRETRAIN = 30

    # Stage 2: Supervised Fine-Tuning (Classifier)
    LR_FINETUNE = 1e-3
    EPOCHS_FINETUNE = 30
    PATIENCE = 5  # Early stopping patience

    # Loss Weights for DAE
    # Weight for Binary Cross Entropy (reconstructing binary feats) vs MSE (continuous)
    BCE_WEIGHT = 1.0
    MSE_WEIGHT = 1.0
