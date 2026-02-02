import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of data loading workers

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Directories
    WORKING_DIR = "./working/idea_6"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Image Resolution: High resolution as requested for fine fracture lines
    IMAGE_SIZE = 384

    # Sequence Length: Number of slices sampled per study (Z-axis uniform sampling)
    # Balanced to fit in memory with EfficientNet-B4 backbone
    SEQ_LEN = 24

    # Input Channels: 3 channels for 2.5D stacking (z-1, z, z+1)
    IN_CHANNELS = 3

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone
    BACKBONE = "efficientnet_b4"
    BACKBONE_PRETRAINED = True

    # Feature Dimensions
    # EfficientNet-B4 usually outputs 1792 features at the final conv layer
    # We project this to HIDDEN_DIM
    HIDDEN_DIM = 512

    # Sequential Context (LSTM)
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.1
    BIDIRECTIONAL = True

    # Transformer Decoder (Set Prediction)
    NUM_QUERIES = 8  # C1, C2, C3, C4, C5, C6, C7, Patient_Overall
    NHEAD = 8
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    ACTIVATION = "relu"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 10

    # Batch Size & Gradient Accumulation
    # Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS
    # BATCH_SIZE is per GPU pass. Kept small due to large 3D volume inputs.
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4  # Simulates batch size of 16

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 5.0

    # Loss Function
    # Weighted Multi-Label Logarithmic Loss
    # Positive weight > 1 to handle class imbalance and improve sensitivity
    POS_WEIGHT = 2.0

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # =========================================================================
    # Inference
    # =========================================================================
    # Threshold for converting probabilities to binary predictions (if needed for metrics)
    THRESHOLD = 0.5
