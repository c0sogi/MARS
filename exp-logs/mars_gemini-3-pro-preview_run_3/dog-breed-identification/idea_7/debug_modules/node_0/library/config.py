import os
import torch


class Config:
    """
    Configuration for the Heterogeneous Ensemble Dog Breed Classification Task.
    Implements the strategy defined in Idea 7:
    - Models: ConvNeXt-Small + Swin-Small
    - Training: 5-Fold CV, LLRD, SWA, Cosine Annealing
    - Data: 224x224, Hard Labels
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs
    DEBUG_SAMPLE_SIZE = 200  # Number of samples if DEBUG is True

    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 7 (Cache and Checkpoints)
    WORK_DIR = "./working/idea_7"
    os.makedirs(WORK_DIR, exist_ok=True)

    OUTPUT_DIR = "./submission"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 224
    NUM_CLASSES = 120
    N_FOLDS = 5
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    # Heterogeneous Ensemble Components
    # 1. ConvNeXt Small (Feature extraction: Texture/Local)
    # 2. Swin Transformer Small (Feature extraction: Shape/Global)
    MODELS = [
        "convnext_small.fb_in22k_ft_in1k",
        "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    ]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 30
    BATCH_SIZE = 32  # Conservative for A100 to fit both models if needed, or per model

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Standard AdamW weight decay

    # Layer-Wise Learning Rate Decay (LLRD)
    # Decay rate for layers as we go deeper into the backbone
    LLRD_DECAY = 0.8

    # Scheduler (Cosine Annealing)
    T_MAX = 30
    MIN_LR = 1e-6

    # ==========================================
    # Stochastic Weight Averaging (SWA)
    # ==========================================
    USE_SWA = True
    SWA_START_EPOCH = 24
    SWA_LR = 5e-5

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("-" * 30)
        print("Configuration:")
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("-" * 30)
