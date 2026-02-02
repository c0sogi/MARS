import os
import torch


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_meta.csv")

    # Image Directories (relative to INPUT_DIR as per metadata generation)
    # The metadata contains 'file_path' which joins these already,
    # but we keep base paths just in case.
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Files
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    EMA_MODEL_PATH = os.path.join(WORKING_DIR, "ema_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    LOG_FILE = os.path.join(WORKING_DIR, "train.log")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Using ConvNeXt-Small as requested for better feature extraction
    MODEL_NAME = "convnext_small.fb_in1k"

    # High resolution input for fine-grained detection
    IMG_SIZE = 384

    # 23 Classes: 0 (Empty) + 22 Species
    NUM_CLASSES = 23

    # Auxiliary Head for "Animal vs Empty" detection
    # 1 output logit for binary classification (BCEWithLogitsLoss)
    NUM_DETECTION_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Seed for reproducibility
    SEED = 42

    # Batch size: 384x384 is memory intensive.
    # A100 40GB can handle ~48-64 depending on mixed precision.
    BATCH_SIZE = 16

    # Number of epochs
    EPOCHS = 10

    # Optimizer settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05

    # Loss Function Weights
    # Total Loss = Species_Loss + LAMBDA_DETECTION * Detection_Loss
    LAMBDA_DETECTION = 1.0  # Balance the auxiliary task

    # Early Stopping
    PATIENCE = 3

    # EMA Decay
    EMA_DECAY = 0.9999

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # 12 vCPUs available

    # ==========================================
    # Class Mapping
    # ==========================================
    ID2NAME = {
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
