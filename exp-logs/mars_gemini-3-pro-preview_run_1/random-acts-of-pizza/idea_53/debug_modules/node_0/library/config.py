import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific directory for this solution's artifacts (caching)
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_53")
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw JSON Files (for text extraction if needed beyond metadata)
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Text Columns
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Semantic Embedding
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

    # TF-IDF & Top-K
    TFIDF_MAX_FEATURES = 5000
    TOP_K_SUBREDDITS = 50

    # -------------------------------------------------------------------------
    # Stream A: Random Forest Hyperparameters
    # -------------------------------------------------------------------------
    RF_N_ESTIMATORS = 500
    RF_CLASS_WEIGHT = "balanced"
    RF_MIN_SAMPLES_LEAF = 1
    RF_N_JOBS = -1  # Use all available cores

    # -------------------------------------------------------------------------
    # Stream B: MLP Hyperparameters
    # -------------------------------------------------------------------------
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT_EMBEDDING = 0.5  # Higher dropout for raw embeddings
    MLP_DROPOUT_DENSE = 0.2  # Standard dropout for dense layers
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-4
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # Early stopping patience

    # -------------------------------------------------------------------------
    # Ensemble Parameters
    # -------------------------------------------------------------------------
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5

    @classmethod
    def print_summary(cls):
        """Prints a summary of the current configuration."""
        print("\n" + "=" * 40)
        print(f"{'CONFIGURATION SUMMARY':^40}")
        print("=" * 40)
        print(f"Device:             {cls.DEVICE}")
        print(f"Random Seed:        {cls.RANDOM_SEED}")
        print(f"Artifact Dir:       {cls.IDEA_DIR}")
        print("-" * 40)
        print(f"RF Estimators:      {cls.RF_N_ESTIMATORS}")
        print(f"TF-IDF Vocab:       {cls.TFIDF_MAX_FEATURES}")
        print("-" * 40)
        print(f"MLP Epochs:         {cls.MLP_EPOCHS}")
        print(f"MLP Patience:       {cls.MLP_PATIENCE}")
        print(f"MLP Dropout (Emb):  {cls.MLP_DROPOUT_EMBEDDING}")
        print("=" * 40 + "\n")
