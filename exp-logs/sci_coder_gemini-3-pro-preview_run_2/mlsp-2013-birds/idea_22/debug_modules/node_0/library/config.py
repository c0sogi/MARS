import os
import torch


class Config:
    # Random Seed
    SEED = 42

    # Data Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_22"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Image Configuration
    # Rectangular resolution: 224 (Freq) x 448 (Time)
    IMG_HEIGHT = 224
    IMG_WIDTH = 448
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    CHANNELS = 3  # Pseudo-RGB

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 25  # Sufficient for convergence with pre-trained models
    LEARNING_RATE = 1e-3
    N_FOLDS = 5

    # Regularization & Augmentation
    MIXUP_ALPHA = 0.4

    # SAM (Sharpness-Aware Minimization) Parameters
    SAM_RHO = 0.05
    SAM_ADAPTIVE = False

    # Distillation Parameters
    DISTILLATION_LAMBDA = 1.0  # Weight for KL Divergence loss

    # Model Architecture Names
    MODEL_RESNET = "resnet18"
    MODEL_EFFICIENTNET = "efficientnet_b0"
    MODEL_DENSENET = "densenet121"

    # Dataset Specifics
    NUM_CLASSES = 19

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50
