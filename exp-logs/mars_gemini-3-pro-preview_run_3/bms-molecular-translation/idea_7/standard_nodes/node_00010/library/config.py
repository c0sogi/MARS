import os
import torch


class Config:
    """
    Global configuration for the InChI prediction task.
    """

    # -------------------------------------------------------------------------
    # Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Artifacts
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "model_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    # Image dimensions: Wide aspect ratio for scan-based encoding
    IMAGE_HEIGHT = 192
    IMAGE_WIDTH = 512

    # Text parameters
    # Max length observed in EDA is 403. Adding margin for SOS/EOS tokens.
    MAX_LENGTH = 450

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Encoder (ResNet + BiLSTM)
    RESNET_ARCH = "resnet18"  # Base CNN
    ENCODER_HIDDEN_DIM = 256  # Dimension for the BiLSTM (output will be 2x this)

    # Decoder (Attention LSTM)
    DECODER_HIDDEN_DIM = 512  # Should match encoder output dim (2 * 256)
    ATTENTION_DIM = 256  # Dimension for attention scoring
    EMBEDDING_DIM = 256  # Dimension for character embeddings
    DROPOUT = 0.5  # Dropout probability

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 128  # Optimized for A100 GPU
    LEARNING_RATE = 1e-4  # Standard starting rate for RMSprop/Adam
    WEIGHT_DECAY = 1e-6
    EPOCHS = 10  # Number of training epochs
    NUM_WORKERS = 4  # Number of dataloader workers

    # Teacher forcing scheduler
    TEACHER_FORCING_RATIO = 0.5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # If True, runs on a small subset of the data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    # Validation frequency
    PRINT_FREQ = 100  # Print training metrics every N batches
