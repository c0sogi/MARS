import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Model Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data & Vocabulary
    # -------------------------------------------------------------------------
    IMG_HEIGHT = 128
    MAX_WIDTH = 1024  # Max width for padding/resizing limit if needed

    # Vocabulary derived from data analysis
    # CTC Loss requires a blank token, usually at index 0
    CHAR_VECTOR = sorted(
        [
            "(",
            ")",
            "+",
            ",",
            "-",
            "/",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "=",
            "B",
            "C",
            "D",
            "F",
            "H",
            "I",
            "N",
            "O",
            "P",
            "S",
            "T",
            "b",
            "c",
            "h",
            "i",
            "l",
            "m",
            "n",
            "r",
            "s",
            "t",
        ]
    )

    VOCAB = ["<blank>"] + CHAR_VECTOR
    VOCAB_SIZE = len(VOCAB)

    CHAR2IDX = {char: idx for idx, char in enumerate(VOCAB)}
    IDX2CHAR = {idx: char for idx, char in enumerate(VOCAB)}

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    CNN_BACKBONE = "resnet18"
    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    PATIENCE = 5  # Early stopping patience
    NUM_WORKERS = 8

    # Gradient clipping
    MAX_NORM = 5.0

    # -------------------------------------------------------------------------
    # Reproducibility & Device
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configuration loaded. Device: {cls.DEVICE}")
        print(f"Working Directory: {cls.WORKING_DIR}")
        print(f"Vocabulary Size: {cls.VOCAB_SIZE}")
