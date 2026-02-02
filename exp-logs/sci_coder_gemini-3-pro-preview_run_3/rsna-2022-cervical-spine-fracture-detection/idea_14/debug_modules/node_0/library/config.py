import os
import torch


class Config:
    """
    Configuration for the Multi-Stage Feature-Fused ConvNeXt MIL Network.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Data Paths ---
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --- Caching ---
    # Directory for storing preprocessed .npy files (2.5D stacks)
    # Logic: If load_cached_data=True, read from here. Else, compute and save here.
    CACHE_DIR = "./working/idea_14"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Output Paths ---
    # The final submission file
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model checkpoints
    MODEL_DIR = "./working"
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.pth")

    # --- Data Processing Hyperparameters ---
    # Input Dimensions
    NUM_SLICES = 64  # Uniform sampling depth
    IMAGE_SIZE = (256, 256)  # H, W (Divisible by 32 for ConvNeXt stages)
    IN_CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)

    # Bone Windowing Settings (Hounsfield Units)
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # --- Model Architecture ---
    BACKBONE = "convnext_tiny"  # LayerNorm-native backbone for stability
    NUM_CLASSES = 8  # C1-C7 (7) + Patient Overall (1)

    # --- Training Hyperparameters ---
    BATCH_SIZE = 8  # Small batch size as per strategy
    EPOCHS = 10  # Number of training epochs
    LEARNING_RATE = 2e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-2  # Regularization
    MAX_GRAD_NORM = 1000  # Gradient clipping threshold

    # Scheduler Settings (Decoupled Cosine Annealing)
    T_MAX_MULTIPLIER = 1.5  # T_max = 1.5 * EPOCHS
    MIN_LR = 1e-6  # Minimum learning rate

    # --- Hardware ---
    NUM_WORKERS = 12  # Number of DataLoader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Debugging ---
    DEBUG = False  # Set True to run on a small subset
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use in debug mode
