import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Directory & File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching and checkpoints
    # Using 'idea_12' as specified in the prompt requirements
    WORKING_DIR = "./working/idea_12"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    IMG_SIZE = 320
    NUM_CLASSES = 4029  # Based on metadata analysis (including new_whale)
    # Note: The actual number of classes will be determined dynamically by the dataset loader,
    # but this is the expected count from the analysis.

    # -------------------------------------------------------------------------
    # Model Parameters
    # -------------------------------------------------------------------------
    EMBEDDING_SIZE = 512
    DROPOUT = 0.0  # Explicitly excluded per strategy

    # ArcFace Hyperparameters
    ARCFACE_S = 30.0
    ARCFACE_M = 0.50

    # Ensemble Configuration
    # Heterogeneous Ensemble: 2x DenseNet121, 2x ResNet50-IBN-a
    # Each model has a unique seed for independent convergence.
    ENSEMBLE_MODELS = [
        {"arch": "densenet121", "seed": 42, "name": "densenet_seed42"},
        {"arch": "densenet121", "seed": 2023, "name": "densenet_seed2023"},
        {"arch": "resnet50_ibn_a", "seed": 101, "name": "resnet_ibn_seed101"},
        {"arch": "resnet50_ibn_a", "seed": 999, "name": "resnet_ibn_seed999"},
    ]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 3e-4  # Conservative LR
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4

    # Regularization
    LABEL_SMOOTHING = 0.1

    # Training Loop
    MAX_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Inference Parameters
    # -------------------------------------------------------------------------
    TTA_FLIP = True  # Test Time Augmentation: Horizontal Flip
    TOP_K = 5
    NEW_WHALE_THRESHOLD = (
        None  # Not strictly used if 'new_whale' is a class, but good for reference
    )

    @classmethod
    def print_config(cls):
        print("=" * 30)
        print("CONFIGURATION")
        print("=" * 30)
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"ArcFace: s={cls.ARCFACE_S}, m={cls.ARCFACE_M}")
        print(f"Label Smoothing: {cls.LABEL_SMOOTHING}")
        print(f"Ensemble Members: {len(cls.ENSEMBLE_MODELS)}")
        for model in cls.ENSEMBLE_MODELS:
            print(f"  - {model['arch']} (Seed: {model['seed']})")
        print("=" * 30)
