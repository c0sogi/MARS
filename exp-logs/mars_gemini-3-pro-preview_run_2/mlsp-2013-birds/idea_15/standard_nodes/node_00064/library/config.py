import os
import torch


class Config:
    """
    Configuration for the Tri-Architecture Multi-Resolution Ensemble pipeline.
    Includes settings for data paths, model architectures, training hyperparameters,
    and augmentation strategies.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and saving checkpoints
    # Using 'idea_15' to isolate this specific experiment
    WORKING_DIR = "./working/idea_15"

    # Output directory for final submission file
    OUTPUT_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Metadata File Paths (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Input Data Source
    # Using Filtered Spectrograms to reduce noise (Lesson 28)
    IMAGE_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "filtered_spectrograms")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_CLASSES = 19
    NUM_WORKERS = 4  # Number of DataLoader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging control
    # Set DEBUG = True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLES = 50

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Iterative Stratified K-Fold (K=5)
    N_FOLDS = 5

    # Training Duration
    # 40 epochs is sufficient given the dataset size and pre-trained models
    EPOCHS = 40

    # Batch Size (A100 GPU allows for larger batches, but 32 is standard for stability)
    BATCH_SIZE = 32

    # Optimization
    # AdamW with Constant Learning Rate (Lesson 36, 40)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Function
    # Use pos_weight to handle class imbalance
    USE_POS_WEIGHT = True

    # ==========================================
    # Augmentation Parameters
    # ==========================================
    # Mixup (Lesson 22)
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4

    # Spectrogram Time-Rolling (Circular Shift)
    # Exploits temporal translation invariance of bird calls
    TIME_ROLL_PROB = 0.5

    # SpecAugment parameters (Frequency and Time masking)
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 40

    # ==========================================
    # Model Architecture & Resolutions
    # ==========================================
    # Multi-Resolution Strategy (Lesson 52):
    # - ResNet18/EffNet use higher resolution (224x448) for detail.
    # - DenseNet uses lower resolution (160x320) for regularization.

    # Resolution Definitions (Freq x Time)
    RES_HIGH = (224, 448)
    RES_LOW = (160, 320)

    # Architecture Configurations
    MODEL_CONFIGS = {
        "resnet18": {
            "model_name": "resnet18",
            "img_size": RES_HIGH,
            "pretrained": True,
        },
        "efficientnet_b0": {
            "model_name": "efficientnet_b0",
            "img_size": RES_HIGH,
            "pretrained": True,
        },
        "densenet121": {
            "model_name": "densenet121",
            "img_size": RES_LOW,
            "pretrained": True,
        },
    }

    # Multi-Sample Dropout Head (Idea 13)
    # Stabilizes convergence by using multiple dropout masks in the final layer
    USE_MULTI_SAMPLE_DROPOUT = True
    DROPOUT_SAMPLES = 5
    DROPOUT_RATE = 0.5


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
