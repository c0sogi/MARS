import os
import torch


class Config:
    # ==============================
    # File Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for the current idea (idea_26)
    # Used to store processed slabs or intermediate artifacts if necessary
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_26")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata CSV Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_FILE = "./submission.csv"

    # ==============================
    # Compute & Reproducibility
    # ==============================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, setting workers to a safe number for data loading
    NUM_WORKERS = 2

    # ==============================
    # Model Hyperparameters
    # ==============================
    # SegFormer MiT-B2 architecture
    ENCODER_NAME = "mit_b2"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 8
    LEARNING_RATE = 6e-5
    EPOCHS = 15

    # ==============================
    # Data & Z-Axis Strategy
    # ==============================
    TILE_SIZE = 512
    STRIDE = 512

    # "Overlapping Thick Slab" parameters
    # The depth of the 3D chunk extracted from the volume
    SLAB_DEPTH = 12

    # Dynamic Safe-View Sampling Range for Training
    # During training, we randomly sample a Z-start index from this range.
    # This ensures the central ink volume (approx slice 32) is always captured
    # but appears in different channels/positions to force translation invariance.
    TRAIN_Z_RANGE = (16, 24)

    # Deterministic Z-Scanning for Inference
    # At test time, we predict on these fixed views and Max-Fuse the probabilities.
    INFERENCE_Z_STARTS = [16, 20, 24]

    # ==============================
    # Evaluation
    # ==============================
    # Probability threshold for binary mask generation
    THRESHOLD = 0.5
