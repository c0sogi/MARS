import os


class Config:
    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test")

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Checkpoint paths
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    # Image resolution for MobileNetV3
    IMAGE_SIZE = (256, 256)

    # Vocabulary from EDA: ()+,-/0123456789=BCDFHINOPSTbchilmnrst
    # Plus special tokens: <PAD>, <SOS>, <EOS>
    VOCAB_CHARS = "()+,-/0123456789=BCDFHINOPSTbchilmnrst"
    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]

    # Full vocabulary list
    VOCAB = SPECIAL_TOKENS + list(VOCAB_CHARS)
    VOCAB_SIZE = len(VOCAB)

    # Token indices
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    # Max sequence length for generation (from EDA max length ~403)
    MAX_SEQ_LEN = 410

    # -------------------------------------------------------------------------
    # Attribute Branch Parameters
    # -------------------------------------------------------------------------
    # Atoms to count for the auxiliary regression task
    # Counts of C, H, O, N, S, P, Halogens (F, Cl, Br, I)
    ATOM_KEYS = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]

    # Attribute dimension: count of specific atoms + 1 for total string length
    ATTRIBUTE_DIM = len(ATOM_KEYS) + 1

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    # MobileNetV3-Small output channels (usually 576 before final classifier,
    # but we will likely use the pooled feature vector size)
    ENCODER_OUT_DIM = 576

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Multi-task loss weight: L_total = L_seq + LAMBDA * L_attr
    LAMBDA_ATTR_LOSS = 0.5

    # Early Stopping
    PATIENCE = 3
