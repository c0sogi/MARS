import os
import torch


class Config:
    """
    Configuration for the Hybrid Cascade Text Normalization model.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- File Paths ---
    # Input Data (Metadata contains the split CSVs)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Sample submission from input directory
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "ru_sample_submission_2.csv")

    # Output Artifacts (Cached Data & Models)
    # N-gram statistics dictionary (saved as numpy object)
    NGRAM_STATS_PATH = os.path.join(WORKING_DIR, "ngram_stats.npy")

    # Tokenizer data
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "char_tokenizer.json")

    # Model Checkpoint
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "neural_normalizer_best.pt")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Heuristic Router Settings ---
    # Regex to route tokens to the Neural Model (if they contain a digit)
    DIGIT_REGEX = r"\d"

    # --- N-gram Memory Settings ---
    NGRAM_ORDER = 3  # Trigram context

    # --- Neural Model Settings (Seq2Seq Transformer) ---
    # Data Processing
    CONTEXT_WINDOW = 1  # Number of tokens to include as context (left and right)
    MAX_INPUT_LEN = 128  # Max length of input character sequence
    MAX_TARGET_LEN = 128  # Max length of target character sequence

    # Architecture
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 4
    N_HEADS = 8
    DROPOUT = 0.1

    # Special Tokens
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # --- Training Hyperparameters ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to an integer (e.g., 50000) to train on a subset for debugging.
    # Set to None to use the full dataset.
    DEBUG_SUBSET_SIZE = None

    NUM_EPOCHS = 10
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    CLIP_GRAD = 1.0

    EARLY_STOPPING_PATIENCE = 3
    NUM_WORKERS = 4

    @classmethod
    def print_summary(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print("CONFIGURATION SUMMARY")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"Debug Subset: {cls.DEBUG_SUBSET_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Model: {cls.N_LAYERS} Layers, {cls.N_HEADS} Heads, {cls.EMBED_DIM} Dim")
        print("=" * 40 + "\n")
