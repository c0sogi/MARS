import os
import torch


class Config:
    """
    Configuration class for the InChI prediction task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # --- General Settings ---
    PROJECT_NAME = "InChI_Prediction_Idea1"
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEBUG_SIZE = 5000  # Number of samples to use in debug mode

    # --- Directory Paths ---
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories (Writeable)
    WORKING_DIR = "./working/idea_1"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --- Data Hyperparameters ---
    IMAGE_SIZE = 384  # Resize dimensions (Square)

    # Max sequence length. EDA max was 403.
    # We add a buffer for special tokens (<SOS>, <EOS>) and potential padding.
    MAX_TEXT_LEN = 410

    # Vocabulary derived from EDA analysis
    # Special tokens (<PAD>, <SOS>, <EOS>, <UNK>) will be handled by the Tokenizer class
    VOCAB_STRING = "()+,-/0123456789=BCDFHINOPSTbchilmnrst"

    # --- Model Architecture (Show and Tell) ---
    # Encoder: MobileNetV3-Large (Efficient CNN)
    ENCODER_NAME = "mobilenetv3_large_100"
    ENCODER_PRETRAINED = True

    # Decoder: LSTM
    EMBEDDING_DIM = 256
    DECODER_HIDDEN_SIZE = 512
    DECODER_LAYERS = 1
    DROPOUT = 0.5

    # --- Training Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_EPOCHS = 15
    BATCH_SIZE = 64  # Adjusted for A100 40GB and 384x384 images
    NUM_WORKERS = 8  # Utilizing available vCPUs

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    CLIP_GRAD_NORM = 5.0

    # Teacher forcing schedule could be implemented, but fixed ratio is a good baseline
    TEACHER_FORCING_RATIO = 0.5

    EARLY_STOPPING_PATIENCE = 3

    # --- Inference Hyperparameters ---
    MAX_PRED_LEN = 410

    @classmethod
    def create_dirs(cls):
        """Creates necessary output directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically create directories when config is imported
Config.create_dirs()
