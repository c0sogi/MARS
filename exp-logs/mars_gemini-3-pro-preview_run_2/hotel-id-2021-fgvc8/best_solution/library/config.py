import os
import torch


class Config:
    """
    Configuration class for Hotel Identification Task (Idea 3).
    Centralizes all hyperparameters, paths, and model settings.
    """

    # =======================
    # General Settings
    # =======================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use when DEBUG is True

    # =======================
    # Compute
    # =======================
    # Use available GPU, fallback to CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of workers for DataLoader (12 vCPUs available)
    NUM_WORKERS = 8

    # =======================
    # Directories & Paths
    # =======================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Create working and submission directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata CSV Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output File Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (for Embeddings)
    GALLERY_EMB_PATH = os.path.join(WORKING_DIR, "gallery_embeddings.npy")
    GALLERY_LABELS_PATH = os.path.join(WORKING_DIR, "gallery_labels.npy")
    QUERY_EMB_PATH = os.path.join(WORKING_DIR, "query_embeddings.npy")

    # =======================
    # Model Architecture
    # =======================
    BACKBONE_NAME = "tf_efficientnet_b4"
    EMBEDDING_DIM = 512
    PRETRAINED = True
    USE_GEM_POOLING = True  # Generalized Mean Pooling

    # Sub-Center ArcFace Head Parameters
    NUM_CLASSES = 7770  # Total unique hotels in training set
    NUM_SUB_CENTERS = 3  # K=3 sub-centers per class
    MARGIN = 0.50
    SCALE = 30.0

    # =======================
    # Data / Preprocessing
    # =======================
    IMAGE_SIZE = 384
    INPUT_SHAPE = (384, 384)
    MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    STD = [0.229, 0.224, 0.225]

    # =======================
    # Training Hyperparameters
    # =======================
    BATCH_SIZE = 24  # Adjusted for EffNet-B4 @ 384x384 on A100 GPU
    EPOCHS = 12  # Total training epochs
    WARMUP_EPOCHS = 1  # Epochs to freeze backbone and train only the head

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler settings (Cosine Annealing)
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_MIN_LR = 1e-6

    # =======================
    # Inference / Retrieval
    # =======================
    KNN_K = 50  # Number of nearest neighbors to retrieve for predictions
