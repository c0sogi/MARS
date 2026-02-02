import os
import torch


class Config:
    """
    Configuration class for Identity-Weighted RoBERTa-Large with SWA.
    Centralizes all hyperparameters, file paths, and hardware settings.
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 1000

    # Directory definitions
    # Using 'idea_6' as the working directory for this specific experiment iteration
    WORKING_DIR = "./working/idea_6"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Paths
    # --------------------------------------------------------------------------
    # Metadata files (contain labels and split info, but no text)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Source text files (referenced by metadata)
    TRAIN_TEXT_SOURCE = os.path.join(INPUT_DIR, "train.csv")
    TEST_TEXT_SOURCE = os.path.join(INPUT_DIR, "test.csv")

    # Submission output
    SUBMISSION_PATH = "./submission/submission.csv"

    # --------------------------------------------------------------------------
    # Model Architecture & Tokenizer
    # --------------------------------------------------------------------------
    MODEL_NAME = "roberta-large"
    TOKENIZER_NAME = "roberta-large"

    # Input sequence length
    MAX_LEN = 512

    # Model dimensions (RoBERTa-Large defaults)
    HIDDEN_SIZE = 1024
    DROPOUT = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Batch size: RoBERTa-Large is memory intensive.
    # With 40GB A100, we can handle ~16-32 depending on mixed precision.
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32

    # Optimization
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # --------------------------------------------------------------------------
    # Bias Mitigation Strategies
    # --------------------------------------------------------------------------
    # 1. Identity-Based Sample Weighting
    # Scalar weight applied to the primary toxicity loss for samples mentioning identities.
    # This forces the model to focus on the "hard" examples (potential bias cases).
    IDENTITY_WEIGHT_BOOST = 5.0

    # 2. Multi-Task Learning
    # Weight for the auxiliary identity prediction head.
    # Kept balanced (0.5) to guide representation without dominating the gradient.
    AUX_LOSS_WEIGHT = 0.5

    # 3. Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 2  # Start averaging weights from this epoch onwards
    SWA_LR = 5e-6  # Learning rate for SWA phase

    # --------------------------------------------------------------------------
    # Column Definitions
    # --------------------------------------------------------------------------
    TARGET_COL = "target"
    TEXT_COL = "comment_text"
    ID_COL = "id"

    # Identity attributes used for auxiliary task and bias metrics
    IDENTITY_COLS = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]
    NUM_AUX_CLASSES = len(IDENTITY_COLS)

    # --------------------------------------------------------------------------
    # Hardware & Performance
    # --------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # --------------------------------------------------------------------------
    # Caching Paths (for preprocessed tensors)
    # --------------------------------------------------------------------------
    # Training Cache
    CACHE_TRAIN_INPUT_IDS = os.path.join(WORKING_DIR, "train_input_ids.npy")
    CACHE_TRAIN_ATTN_MASKS = os.path.join(WORKING_DIR, "train_attn_masks.npy")
    CACHE_TRAIN_TARGETS = os.path.join(WORKING_DIR, "train_targets.npy")
    CACHE_TRAIN_AUX_TARGETS = os.path.join(WORKING_DIR, "train_aux_targets.npy")
    CACHE_TRAIN_SAMPLE_WEIGHTS = os.path.join(WORKING_DIR, "train_sample_weights.npy")

    # Validation Cache
    CACHE_VAL_INPUT_IDS = os.path.join(WORKING_DIR, "val_input_ids.npy")
    CACHE_VAL_ATTN_MASKS = os.path.join(WORKING_DIR, "val_attn_masks.npy")
    CACHE_VAL_TARGETS = os.path.join(WORKING_DIR, "val_targets.npy")
    CACHE_VAL_AUX_TARGETS = os.path.join(WORKING_DIR, "val_aux_targets.npy")
    CACHE_VAL_IDS = os.path.join(
        WORKING_DIR, "val_ids.npy"
    )  # Needed for bias metric calculation

    # Test Cache
    CACHE_TEST_INPUT_IDS = os.path.join(WORKING_DIR, "test_input_ids.npy")
    CACHE_TEST_ATTN_MASKS = os.path.join(WORKING_DIR, "test_attn_masks.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model Checkpoints
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SWA_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "swa_model.pth")
