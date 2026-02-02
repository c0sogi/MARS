import os
import torch


class Config:
    """
    Configuration for Idea 6: LinkNet with Additive Skip Connections.
    Centralizes all parameters for data, model, training, and inference.
    """

    # -----------------------------
    # General & Compute
    # -----------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs for data loading
    NUM_WORKERS = os.cpu_count() or 4

    # Debugging flags
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use when DEBUG is True

    # -----------------------------
    # Paths & Directories
    # -----------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for Idea 6 (cache and model checkpoints)
    WORKING_DIR = "./working/idea_6"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model checkpoint path
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -----------------------------
    # Data Parameters
    # -----------------------------
    IMG_SIZE = 256  # Resizing resolution
    IN_CHANNELS = 3  # 2.5D Input: Slice i-1, Slice i, Slice i+1
    NUM_CLASSES = 3  # Large Bowel, Small Bowel, Stomach
    CLASS_LABELS = ["large_bowel", "small_bowel", "stomach"]

    # Normalization
    NORM_MIN = 0.0
    NORM_MAX = 1.0

    # -----------------------------
    # Model Architecture
    # -----------------------------
    MODEL_NAME = "LinkNet"
    BACKBONE = "resnet18"
    PRETRAINED = True

    # -----------------------------
    # Training Hyperparameters
    # -----------------------------
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Loss Function Weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # -----------------------------
    # Post-Processing & Inference
    # -----------------------------
    PRED_THRESHOLD = 0.5
    USE_3D_CONNECTED_COMPONENTS = True  # Keep only largest 3D component per class
