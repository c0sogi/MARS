import os
import torch


class Config:
    """
    Central configuration for the Heterogeneous Ensemble with Multi-Axis Attention
    and Stochastic Weight Averaging (SWA) experiment.
    """

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # General Setup & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging Flags
    # Set DEBUG to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # -------------------------------------------------------------------------
    # Stochastic Weight Averaging (SWA) Settings
    # -------------------------------------------------------------------------
    # SWA helps find a flatter minimum in the loss landscape for better generalization
    USE_SWA = True
    SWA_START_EPOCH = 8  # Start collecting SWA weights after this epoch (0-indexed)
    SWA_LR = 1e-5  # Constant learning rate during the SWA phase

    # -------------------------------------------------------------------------
    # Multi-Sample Dropout Head Settings
    # -------------------------------------------------------------------------
    # Creates an internal ensemble within the model to reduce log loss
    USE_MULTI_SAMPLE_DROPOUT = True
    DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

    # -------------------------------------------------------------------------
    # Data Augmentation
    # -------------------------------------------------------------------------
    # Context-Preservation and Photometric Noise
    AUG_SCALE = (0.8, 1.0)
    COLOR_JITTER_INTENSITY = 0.2

    # -------------------------------------------------------------------------
    # Model Specifications (Triple Heterogeneous Ensemble)
    # -------------------------------------------------------------------------
    # Defines the specific architectures and their decoupled resolutions.
    # 1. ResNet-50: Standard CNN anchor (256x256)
    # 2. ConvNeXt-Small: Modern CNN (256x256)
    # 3. MaxViT-Tiny: Multi-Axis Transformer (224x224)

    MODEL_SPECS = {
        "resnet50": {
            "timm_name": "resnet50.a1_in1k",  # ResNet50 trained with improved recipe (V2 equivalent)
            "img_size": 256,
            "batch_size": 64,
        },
        "convnext_small": {
            "timm_name": "convnext_small.fb_in1k",
            "img_size": 256,
            "batch_size": 32,
        },
        "maxvit_tiny": {
            "timm_name": "maxvit_tiny_tf_224.in1k",
            "img_size": 224,
            "batch_size": 32,
        },
    }

    # -------------------------------------------------------------------------
    # Inference Settings
    # -------------------------------------------------------------------------
    TTA_FLIP = True  # Enable Horizontal Flip Test Time Augmentation
