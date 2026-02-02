import os
import torch


class Config:
    """
    Configuration for the Calibrated 2.5D Multi-Level Sequence Network (Idea 12).
    Centralizes control for hyperparameters, paths, and model settings.
    """

    # --- General Experiment Settings ---
    PROJECT_NAME = "cervical-spine-fracture-detection"
    EXP_NAME = "idea_12"
    SEED = 42

    # Debugging: Set to True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Output directory for this specific experiment (checkpoints, cache, logs)
    # Ensures directory safety as required
    OUTPUT_DIR = os.path.join(WORKING_DIR, EXP_NAME)

    # Pre-generated metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image source directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # --- Data Preprocessing & Input ---
    # 2.5D Stacking: Input channels = 3 (slices z-1, z, z+1)
    IN_CHANNELS = 3

    # Image Resolution: Reduced to 256 to prevent OOM. Cite debug_lesson_1.
    IMAGE_SIZE = (256, 256)

    # Sequence Length: High density sampling (96 slices) to capture fine fractures
    SEQ_LEN = 96

    # --- Model Architecture ---
    # Backbone: EfficientNet-B4 (timm implementation)
    BACKBONE = "tf_efficientnet_b4.ns_jft_in1k"

    # Sequence Modeling
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True

    # Classification Heads: 7 Vertebrae (C1-C7) + 1 Patient Overall
    NUM_CLASSES = 8
    DROPOUT = 0.2

    # --- Training Hyperparameters ---
    EPOCHS = 10

    # Batch Size: Reduced to 1 to prevent OOM (Effective batch = 1 * 96 = 96 images). Cite debug_lesson_1.
    BATCH_SIZE = 1

    # Gradient Accumulation: Increased to 16 to compensate for small batch size
    ACCUMULATION_STEPS = 16

    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 1000.0

    # --- Loss Function ---
    # Calibration Requirement: No positive class weighting to ensure calibrated probabilities.
    POS_WEIGHT = 1.0

    # --- Compute Resources ---
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def create_dirs(cls):
        """
        Ensures the output directory exists.
        This handles the 'Directory Safety' requirement.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# Initialize directories upon module import
Config.create_dirs()
