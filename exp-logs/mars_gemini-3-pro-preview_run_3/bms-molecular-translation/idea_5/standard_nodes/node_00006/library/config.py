import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility & Debugging
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to train on a small subset for verification
    DEBUG_SUBSET_SIZE = 2000

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Write access for checkpoints, cache)
    WORKING_DIR = "./working/idea_5"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output Paths
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "resnet_tcn_best.pth")

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 256
    # Max InChI length observed is 403. We set 450 to include buffer for SOS/EOS
    # and potential longer sequences in the test set.
    MAX_LEN = 450

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Encoder (ResNet-34)
    ENCODER_NAME = "resnet34"
    ENCODER_PRETRAINED = True
    # ResNet34 global pool output dimension is 512
    ENCODER_DIM = 512

    # Decoder (Temporal Convolutional Network)
    EMBEDDING_DIM = 256
    # TCN Architecture:
    # We use 8 layers with kernel size 3.
    # Receptive Field = 1 + 2*(3-1)*(2^8 - 1) = 1 + 4*255 = 1021.
    # This RF > MAX_LEN (450), ensuring full context coverage.
    TCN_NUM_CHANNELS = [512] * 8
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 128
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    NUM_WORKERS = 4
    PATIENCE = 5  # Early stopping patience

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def print_config():
        print("=" * 40)
        print("MODEL CONFIGURATION (ResNet-TCN)")
        print("=" * 40)
        for k, v in Config.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<25}: {v}")
        print("=" * 40)
