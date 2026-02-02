import os
import torch


class Config:
    """
    Configuration for Hybrid Multi-View Similarity Ensemble.
    Centralizes file paths, hyperparameters, and compute settings.
    """

    # ==========================================
    # Reproducibility & Compute
    # ==========================================
    SEED = 42
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Data Files
    PATH_ARTICLES = os.path.join(INPUT_DIR, "articles.csv")
    PATH_CUSTOMERS = os.path.join(INPUT_DIR, "customers.csv")
    PATH_IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    PATH_SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Generated Metadata (Splits)
    PATH_TRAIN = os.path.join(METADATA_DIR, "train.csv")
    PATH_VAL = os.path.join(METADATA_DIR, "val.csv")
    PATH_TEST = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission
    PATH_SUBMISSION = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Intermediate Artifacts)
    # We use .parquet for DataFrames and .npz for sparse matrices
    CACHE_IMAGE_EMBEDDINGS = os.path.join(WORKING_DIR, "image_embeddings.parquet")
    CACHE_SIM_VISUAL = os.path.join(WORKING_DIR, "similarity_visual.npz")
    CACHE_SIM_BEHAVIOR = os.path.join(WORKING_DIR, "similarity_behavior.npz")
    CACHE_USER_HISTORY = os.path.join(WORKING_DIR, "user_history.parquet")
    CACHE_GLOBAL_TRENDS = os.path.join(WORKING_DIR, "global_trends.parquet")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Temporal Windowing: Use only the last 5 weeks of data for training
    TRAIN_WEEKS = 5

    # Debugging: Set to True to use a smaller subset for rapid iteration
    DEBUG = False
    DEBUG_SAMPLES = 50000

    # ==========================================
    # Visual Model Hyperparameters
    # ==========================================
    # Model Architecture (available in torchvision/timm)
    VISUAL_MODEL_NAME = "resnet50"

    # Image Preprocessing
    IMAGE_SIZE = (224, 224)
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    # Inference Batch Size (A100 40GB can handle large batches)
    BATCH_SIZE = 256

    # ==========================================
    # Ensemble & Similarity Hyperparameters
    # ==========================================
    # Number of neighbors to keep in sparse similarity matrices
    TOP_K_SIMILAR = 20

    # Number of predictions per customer
    TOP_N_PREDICTIONS = 12

    # Default Ensemble Weights (Linear Combination)
    # Score = alpha*Repurchase + beta*Behavioral + gamma*Visual + delta*Trend
    # These serve as initial values or fallbacks if tuning is skipped
    WEIGHT_ALPHA = 1.0  # Repurchase (Habit)
    WEIGHT_BETA = 0.6  # Behavioral (Co-occurrence)
    WEIGHT_GAMMA = 0.4  # Visual (Content-based)
    WEIGHT_DELTA = 0.2  # Global Trend (Popularity)

    @classmethod
    def setup_directories(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
