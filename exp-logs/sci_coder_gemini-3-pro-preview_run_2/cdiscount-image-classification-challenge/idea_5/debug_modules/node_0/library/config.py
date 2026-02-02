import os


class Config:
    """
    Centralized configuration for the Product Categorization task.
    Includes file paths, model hyperparameters, and training settings.
    """

    # ==== Directories ====
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Create writable directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==== File Paths ====
    # Raw Data
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")

    # Metadata (Offsets and Labels)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

    # ==== Data Specifications ====
    IMG_SIZE = 224  # Resizing to 224x224 for ResNet-34 pre-trained weights
    NUM_CLASSES = 5270  # Total number of product categories
    CHANNELS = 3

    # ==== Compute Environment ====
    NUM_WORKERS = 12  # Utilizing all 12 vCPUs for data loading
    PIN_MEMORY = True  # Faster transfer to GPU
    DEVICE = "cuda"  # Assuming GPU availability as per spec

    # ==== Training Hyperparameters ====
    # Batch size optimized for A100 40GB with ResNet-34 @ 224x224
    BATCH_SIZE = 512

    # Training duration limited to 2 epochs to manage 24h runtime on 12M+ images
    EPOCHS = 2

    # Learning Rate (Max LR for OneCycleLR)
    LR = 0.01

    # Regularization
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    # Reproducibility
    SEED = 42
