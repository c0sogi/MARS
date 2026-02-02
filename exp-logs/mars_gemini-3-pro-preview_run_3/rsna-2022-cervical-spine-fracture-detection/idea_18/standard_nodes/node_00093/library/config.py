import os
import torch


class Config:
    # --- General Configuration ---
    DEBUG = False
    SEED = 42
    NUM_WORKERS = 12  # Matches available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Caching ---
    # Directory for storing processed 2.5D stacks or features
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_18")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Model Architecture ---
    BACKBONE = "convnext_tiny"
    PRETRAINED = True
    IN_CHANNELS = 3  # 2.5D Stacking: slices [z-1, z, z+1]
    NUM_CLASSES = 8  # Targets: C1, C2, C3, C4, C5, C6, C7, patient_overall

    # --- Data Preprocessing ---
    IMAGE_SIZE = (224, 224)
    SEQ_LENGTH = 64  # Number of slices sampled per volume

    # Bone Windowing Parameters (HU)
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # Normalization (ImageNet Statistics)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # --- Training Hyperparameters ---
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Scheduler Settings (Cosine Annealing)
    # T_max will be calculated as EPOCHS * T_MAX_COEF
    T_MAX_COEF = 1.5
    MIN_LR = 1e-6

    # --- Targets ---
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # --- Debugging Control ---
    # If not None, limits the number of samples loaded for train/val
    DEBUG_DATA_SIZE = None

    @classmethod
    def set_debug_mode(cls, debug=True, data_size=50, epochs=2):
        """
        Activates debug mode to reduce runtime during development.
        """
        cls.DEBUG = debug
        if debug:
            cls.DEBUG_DATA_SIZE = data_size
            cls.EPOCHS = epochs
            print(
                f"[Config] Debug mode ENABLED. Data Size: {data_size}, Epochs: {epochs}"
            )
        else:
            cls.DEBUG_DATA_SIZE = None
            print("[Config] Debug mode DISABLED.")
