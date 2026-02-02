import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALID_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files (for text extraction)
    TRAIN_TEXT_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_TEXT_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = "./submission/submission.csv"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npy")
    VALID_CACHE_PATH = os.path.join(WORKING_DIR, "valid_data.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npy")
    TOKENIZER_CACHE_DIR = os.path.join(WORKING_DIR, "tokenizer")

    # --------------------------------------------------------------------------
    # Model & Tokenizer
    # --------------------------------------------------------------------------
    MODEL_NAME = "roberta-base"

    # Max length: EDA showed max word length ~315.
    # Subword tokenization increases count. 400 is a safe buffer while less than 512.
    MAX_LEN = 400

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    EPOCHS = 3

    # Batch sizes adapted for A100 40GB
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64

    # Optimizer & Scheduler
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    # Multi-Task Loss Weights
    # High weight for identity to force disentanglement (Strategy Idea 3)
    IDENTITY_WEIGHT = 0.8
    TOXICITY_WEIGHT = (
        1.0  # Usually kept at 1, relative importance handled by IDENTITY_WEIGHT
    )

    # Regularization
    DROPOUT = 0.2
    SPATIAL_DROPOUT = 0.1  # Specific to the custom head

    # --------------------------------------------------------------------------
    # Hardware
    # --------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # 12 vCPUs available

    # --------------------------------------------------------------------------
    # Data Columns
    # --------------------------------------------------------------------------
    TARGET_COL = "target"

    # Identity attributes used for the auxiliary loss and bias metrics
    IDENTITY_COLUMNS = [
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

    # Auxiliary toxicity subtypes (available in data, can be used for analysis)
    AUX_COLUMNS = [
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]
