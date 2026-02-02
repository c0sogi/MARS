import os
import torch


class Config:
    # ==========================================
    # System & Hardware
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 10  # Increased to feed larger batch size

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Working directory for Idea 5
    WORKING_DIR = "./working/idea_5"
    CHECKPOINT_DIR = WORKING_DIR
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    EMA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "ema_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "convnext_small.fb_in1k"
    NUM_CLASSES = 23
    INPUT_SIZE = 224

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 10  # Sufficient given the dataset size and pretraining
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # EMA parameters
    USE_EMA = True
    EMA_DECAY = 0.9999

    # Loss parameters
    FOCAL_LOSS_GAMMA = 2.0

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of samples to use if DEBUG is True

    # ==========================================
    # Label Mapping
    # ==========================================
    ID2LABEL = {
        0: "empty",
        1: "deer",
        2: "moose",
        3: "squirrel",
        4: "rodent",
        5: "small_mammal",
        6: "elk",
        7: "pronghorn_antelope",
        8: "rabbit",
        9: "bighorn_sheep",
        10: "fox",
        11: "coyote",
        12: "black_bear",
        13: "raccoon",
        14: "skunk",
        15: "wolf",
        16: "bobcat",
        17: "cat",
        18: "dog",
        19: "opossum",
        20: "bison",
        21: "mountain_goat",
        22: "mountain_lion",
    }

    LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# Create necessary directories
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
