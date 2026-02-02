import os
import torch


class Config:
    # Random Seed for Reproducibility
    SEED = 42

    # Hardware Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Data Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory for Caching and Outputs
    # Specific to this experiment iteration (idea_14)
    WORKING_DIR = "./working/idea_14"
    PREPROCESSED_DIR = "./working/idea_2/processed_images"
    CACHE_DIR = "./working/idea_14"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Image Configuration
    # Input: 3 Channels (Image + Age Broadcast + Implant Broadcast)
    IMG_SIZE = (768, 768)
    IN_CHANNELS = 3

    # Model Architecture
    BACKBONE = "tf_efficientnet_b2_ns"  # EfficientNet-B2 NoisyStudent

    # Training Hyperparameters
    # Batch size selected for A100 with 768x768 Siamese inputs
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 10

    # Class Imbalance Handling
    # Calculated based on approx 1:47 positive ratio
    POS_WEIGHT = 47.0

    # Debugging / Development Flags
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

    @staticmethod
    def print_config():
        print("=" * 30)
        print("CONFIGURATION")
        print("=" * 30)
        print(f"Device: {Config.DEVICE}")
        print(f"Image Size: {Config.IMG_SIZE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Positive Weight: {Config.POS_WEIGHT}")
        print(f"Backbone: {Config.BACKBONE}")
        print(f"Debug Mode: {Config.DEBUG}")
        print("=" * 30)
