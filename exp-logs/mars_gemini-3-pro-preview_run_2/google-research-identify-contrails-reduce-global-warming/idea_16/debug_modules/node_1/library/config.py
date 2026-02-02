import os
import torch


class Config:
    """
    Configuration for the Contrail Identification Task.
    Implements the 'Isotropic Large-Kernel ConvNeXt U-Net with Decoupled Spatiotemporal Input and Pyramid Fusion' strategy.
    """

    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "idea_16"

    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Model Artifacts
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Directory (for deterministic data processing)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # Data Parameters
    # ==========================================
    IMG_SIZE = 256

    # Temporal sequence details
    # Total frames = n_times_before + n_times_after + 1 = 8
    # Labeled frame is at index n_times_before (index 4)
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3

    # Bands required:
    # Ash Scheme uses: 11, 14, 15
    # Temporal Difference uses: 11, 14, 15
    REQUIRED_BANDS = [11, 14, 15]

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "convnext_tiny"
    PRETRAINED = True

    # Input Channels: 6
    # Channels 1-3: Ash False Color (Static t=4)
    # Channels 4-6: Raw Difference (t=4 - t=3) for Bands 11, 14, 15
    IN_CHANNELS = 6
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Loss & Metrics
    # ==========================================
    # Hybrid Loss: BCE + BatchDice
    LOSS_BCE_WEIGHT = 1.0
    LOSS_DICE_WEIGHT = 1.0
    SMOOTH = 1e-6  # For Dice stability

    # Post-processing
    THRESHOLD = 0.5

    # ==========================================
    # Augmentation & Inference
    # ==========================================
    # Affine only: Rotation, Scale, Shift, Flip
    # Elastic/Grid distortions are explicitly excluded
    AUG_ROTATION = 15  # degrees
    AUG_SCALE = (0.9, 1.1)
    AUG_SHIFT = 0.05

    # Test Time Augmentation (Flip/Rotate)
    TTA_ENABLED = True

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to integer (e.g., 1000) to limit dataset size for quick testing
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None
    DEBUG = False

    @classmethod
    def print_config(cls):
        print(f"{'='*20} CONFIGURATION {'='*20}")
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<25}: {v}")
        print("=" * 55)
