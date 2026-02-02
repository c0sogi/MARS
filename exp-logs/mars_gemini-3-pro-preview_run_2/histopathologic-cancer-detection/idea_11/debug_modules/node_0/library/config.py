import os
import torch


class Config:
    """
    Centralized configuration for the Memory-Resident Homogeneous Bagged Ensemble.
    Defines paths, hardware settings, model architecture, training hyperparameters,
    and augmentation strategies.
    """

    # --- Project & Paths ---
    PROJECT_NAME = "idea_11"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{PROJECT_NAME}"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    for d in [CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # --- Hardware & Compute ---
    # Use A100 GPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available
    NUM_WORKERS = 12
    # Pin memory for faster host-to-device transfer
    PIN_MEMORY = True

    # --- Data Pipeline ---
    # Input image size
    IMG_SIZE = 96
    # Contextual Crop size (center 64x64)
    CROP_SIZE = 64
    # Load entire dataset to RAM for high throughput
    LOAD_TO_RAM = True
    # Dataset specific statistics (from EDA)
    DATASET_MEAN = [0.7035, 0.5476, 0.6975]
    DATASET_STD = [0.2388, 0.2821, 0.2159]

    # --- Model Architecture ---
    # ConvNeXt-Tiny backbone
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    NUM_CLASSES = 1
    # Head configuration: GAP + LayerNorm, no dropout
    USE_GAP = True
    USE_LAYERNORM_HEAD = True
    HEAD_DROPOUT = 0.0
    # Stochastic Depth (DropPath) in backbone
    DROP_PATH_RATE = 0.1

    # --- Training Strategy ---
    SEED = 42
    # Homogeneous Bagged Ensemble: 2 runs of 5-fold CV = 10 models
    NUM_FOLDS = 5
    NUM_RUNS = 2

    # Training Loop
    EPOCHS = 30
    # A100 40GB allows large batch size for tiny model
    BATCH_SIZE = 256

    # Optimization
    OPTIMIZER = "AdamW"
    LR = 1e-3
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.05
    SCHEDULER = "CosineAnnealingLR"
    WARMUP_EPOCHS = 3

    # --- Regularization ---
    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # Exponential Moving Average
    USE_EMA = True
    EMA_DECAY = 0.995

    # --- Inference ---
    # 8-view TTA (Dihedral: 4 rotations * 2 flips)
    TTA_VIEWS = 8

    # --- Debugging ---
    # Set to True to run on a small subset for quick verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"\n[Config] {cls.PROJECT_NAME}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Model: {cls.MODEL_NAME} (Pretrained={cls.PRETRAINED})")
        print(
            f"  Input: {cls.IMG_SIZE}x{cls.IMG_SIZE} -> Crop: {cls.CROP_SIZE}x{cls.CROP_SIZE}"
        )
        print(
            f"  Ensemble: {cls.NUM_RUNS} runs x {cls.NUM_FOLDS} folds = {cls.NUM_RUNS * cls.NUM_FOLDS} models"
        )
        print(f"  Training: {cls.EPOCHS} epochs, Batch Size {cls.BATCH_SIZE}")
        print(
            f"  Regularization: Mixup={cls.MIXUP_ALPHA}, WD={cls.WEIGHT_DECAY}, DropPath={cls.DROP_PATH_RATE}"
        )
        print(f"  EMA: {cls.USE_EMA} (decay={cls.EMA_DECAY})")
        print(f"  Data: Load to RAM={cls.LOAD_TO_RAM}")
        print(f"  Output Dir: {cls.WORKING_DIR}\n")
