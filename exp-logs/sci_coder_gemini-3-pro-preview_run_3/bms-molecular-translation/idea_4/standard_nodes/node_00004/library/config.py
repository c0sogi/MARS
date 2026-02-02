import os
import torch


class Config:
    # --- General ---
    DEBUG = False  # Set to True for quick debugging with a small subset
    SEED = 42
    NUM_WORKERS = 4  # Number of data loading workers
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    MODEL_PATH = os.path.join(WORKING_DIR, "efficientnet_gru_best.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # --- Data ---
    IMAGE_SIZE = 256  # 256x256 resolution
    # Standard ImageNet normalization stats
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # Text / Vocabulary
    MAX_TEXT_LEN = 275  # Covers >99% of InChI strings based on EDA
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    # --- Model Architecture ---
    ENCODER_NAME = "efficientnet_b0"
    ENCODER_DIM = 1280  # Output feature dimension for EfficientNet-B0
    EMBED_DIM = 256
    DECODER_DIM = 512
    ATTENTION_DIM = 256
    DROPOUT = 0.5

    # --- Training ---
    BATCH_SIZE = 128  # A100 can handle larger batches for this model size
    EPOCHS = 6  # Sufficient for convergence given the dataset size
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    CLIP_GRAD = 5.0  # Gradient clipping value
    TEACHER_FORCING_RATIO = 1.0  # Fixed teacher forcing for stability

    # Scheduler
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # --- Debugging ---
    # If DEBUG is True, limit data to this amount
    DEBUG_SAMPLE_SIZE = 1000
