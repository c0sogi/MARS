import os
import torch


class Config:
    """
    Configuration module for Cassava Leaf Disease Classification.
    Implements a Dynamic Fidelity Curriculum using ConvNeXt Small.
    """

    # ====================================================
    # General & Paths
    # ====================================================
    PROJECT_NAME = "Cassava-Leaf-Disease-Classification"
    SEED = 42
    DEBUG = False  # Switch to True for fast debugging runs

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    OUTPUT_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    LABEL_MAP_PATH = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ====================================================
    # Compute Environment
    # ====================================================
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ====================================================
    # Model Architecture
    # ====================================================
    # Using ConvNeXt Small pretrained on ImageNet-1k
    MODEL_NAME = "convnext_small.fb_in1k"
    NUM_CLASSES = 5

    # Regularization
    DROP_PATH_RATE = 0.4  # Stochastic Depth rate

    # Exponential Moving Average
    USE_EMA = True
    EMA_DECAY = 0.9999

    # ====================================================
    # Training Strategy: 5-Fold CV
    # ====================================================
    N_FOLDS = 5

    # ====================================================
    # Two-Phase Dynamic Curriculum
    # ====================================================

    # Phase 1: Coarse Feature Learning
    # Goal: Learn global features rapidly with strong regularization
    P1_IMG_SIZE = 224
    P1_EPOCHS = 12
    P1_BATCH_SIZE = 32
    P1_ACCUM_ITER = 1  # Effective Batch Size = 32
    P1_MIXUP_PROB = 0.5
    P1_LR_MAX = 2e-4

    # Phase 2: Fine-Grained Refinement
    # Goal: Resolve fine details with higher resolution and no interpolation noise
    P2_IMG_SIZE = 384
    P2_EPOCHS = 8
    P2_BATCH_SIZE = 32
    P2_ACCUM_ITER = 1
    P2_MIXUP_PROB = 0.0
    P2_LR_MAX = 5e-5  # Lower LR for fine-tuning
    P2_LABEL_SMOOTHING = 0.1

    # ====================================================
    # Optimization
    # ====================================================
    WEIGHT_DECAY = 0.05
    MIN_LR = 1e-6
    SCHEDULER_TYPE = "cosine"

    # ====================================================
    # Inference
    # ====================================================
    TTA_FLIP = True  # Enable Horizontal Flip Test Time Augmentation

    @classmethod
    def setup(cls, debug=False):
        """
        Initializes the configuration.
        Creates necessary directories and adjusts parameters if in debug mode.
        """
        # Ensure output directories exist
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        if debug:
            cls.DEBUG = True
            cls.P1_EPOCHS = 1
            cls.P2_EPOCHS = 1
            cls.N_FOLDS = 2
            print(f"[{cls.__name__}] Debug mode enabled: Reduced epochs and folds.")

        print(f"[{cls.__name__}] Output Directory: {cls.OUTPUT_DIR}")
        print(f"[{cls.__name__}] Device: {cls.DEVICE}")
