import os


class Config:
    # -------------------------------------------------------------------------
    # 1. Paths & Directories
    # -------------------------------------------------------------------------
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image directories (relative to input)
    TRAIN_IMG_DIR = INPUT_DIR  # Metadata file_path includes 'train/...'
    TEST_IMG_DIR = INPUT_DIR  # Metadata file_path includes 'test/...'

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # -------------------------------------------------------------------------
    # 2. Data Configuration
    # -------------------------------------------------------------------------
    # Image parameters
    IMAGE_SIZE = (256, 256)  # Height, Width
    INPUT_CHANNELS = 1  # EDA shows grayscale images

    # Text parameters
    # Vocabulary derived from EDA
    # Special tokens
    TOKEN_PAD = "<PAD>"
    TOKEN_SOS = "<SOS>"
    TOKEN_EOS = "<EOS>"

    SPECIAL_TOKENS = [TOKEN_PAD, TOKEN_SOS, TOKEN_EOS]

    # Unique characters from EDA
    CHAR_VOCAB = [
        "(",
        ")",
        "+",
        ",",
        "-",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "=",
        "B",
        "C",
        "D",
        "F",
        "H",
        "I",
        "N",
        "O",
        "P",
        "S",
        "T",
        "b",
        "c",
        "h",
        "i",
        "l",
        "m",
        "n",
        "r",
        "s",
        "t",
    ]

    # Full vocabulary list
    VOCABULARY = SPECIAL_TOKENS + CHAR_VOCAB

    # Mappings
    CHAR2IDX = {char: idx for idx, char in enumerate(VOCABULARY)}
    IDX2CHAR = {idx: char for idx, char in enumerate(VOCABULARY)}

    PAD_IDX = CHAR2IDX[TOKEN_PAD]
    SOS_IDX = CHAR2IDX[TOKEN_SOS]
    EOS_IDX = CHAR2IDX[TOKEN_EOS]

    VOCAB_SIZE = len(VOCABULARY)

    # Sequence length (EDA max was 403, adding buffer)
    MAX_LEN = 450

    # -------------------------------------------------------------------------
    # 3. Model Hyperparameters
    # -------------------------------------------------------------------------
    # Encoder (CNN)
    ENCODER_NAME = "resnet18"
    ENCODER_PRETRAINED = True

    # Decoder (Transformer)
    D_MODEL = 256
    NHEAD = 4
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # 4. Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    EPOCHS = 15
    PATIENCE = 3  # Early stopping patience

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of samples to use if DEBUG is True

    @classmethod
    def print_config(cls):
        """Prints the configuration settings."""
        print("=" * 40)
        print("CONFIGURATION")
        print("=" * 40)
        print(f"Working Directory: {cls.WORKING_DIR}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Vocab Size: {cls.VOCAB_SIZE}")
        print(f"Max Sequence Length: {cls.MAX_LEN}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("=" * 40)
