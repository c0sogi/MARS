import os


class Config:
    """
    Configuration class for the Hierarchical Multi-Task ResNet-50 solution.
    Centralizes file paths, model parameters, and training hyperparameters.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================================
    # Directories & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Source Data
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")

    # Metadata Indices (Generated previously)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "resnet50_hierarchical_best.pth")

    # ==========================================
    # Data Parameters
    # ==========================================
    # Image dimensions based on data analysis (180x180)
    IMG_SIZE = 180

    # Hierarchical Class Counts (Derived from Data Analysis)
    # Level 1: Broadest category (e.g., SPORT)
    NUM_CLASSES_L1 = 49
    # Level 2: Intermediate category (e.g., CYCLES)
    NUM_CLASSES_L2 = 483
    # Level 3: Fine-grained target category (e.g., VELO DE VILLE)
    NUM_CLASSES_L3 = 5270

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "resnet50"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch size scaled for A100 GPU (40GB VRAM)
    BATCH_SIZE = 512

    # Maximize CPU utilization for data loading
    NUM_WORKERS = 12

    # Training duration (short duration with aggressive schedule)
    EPOCHS = 2

    # Learning Rate (Max LR for OneCycleLR)
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-4

    # ==========================================
    # Loss Function Configuration
    # ==========================================
    # Hierarchical Loss Weights: L_total = L_fine + lambda1*L_coarse + lambda2*L_inter
    LAMBDA_L1 = 0.10  # Weight for Level 1 auxiliary task
    LAMBDA_L2 = 0.25  # Weight for Level 2 auxiliary task

    # Label Smoothing for the fine-grained target to prevent overfitting on noisy classes
    LABEL_SMOOTHING = 0.1

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
