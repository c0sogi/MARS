import os
import torch


class Config:
    """
    Configuration class for the Retrieval-Augmented Hybrid Cascade Text Normalization model.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Development
    # Set MAX_TRAIN_SAMPLES to an integer (e.g., 50000) to speed up development
    DEBUG = False
    MAX_TRAIN_SAMPLES = None

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure these match the provided metadata structure
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Caching & Artifact Paths
    # ==========================================
    # Subdirectories
    STATS_DIR = os.path.join(WORKING_DIR, "stats")
    PROCESSED_DIR = os.path.join(WORKING_DIR, "processed")

    # Symbolic Statistics (Parquet format required)
    TRIGRAM_STATS_PATH = os.path.join(STATS_DIR, "trigram_stats.parquet")
    BIGRAM_LEFT_STATS_PATH = os.path.join(STATS_DIR, "bigram_left_stats.parquet")
    BIGRAM_RIGHT_STATS_PATH = os.path.join(STATS_DIR, "bigram_right_stats.parquet")
    UNIGRAM_STATS_PATH = os.path.join(STATS_DIR, "unigram_stats.parquet")

    # Retrieval & Model Artifacts
    # Note: Using generic extensions, implementation handles format (e.g., joblib, bin)
    TFIDF_MODEL_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    KNN_INDEX_PATH = os.path.join(WORKING_DIR, "knn_index.bin")
    HARD_SAMPLES_PATH = os.path.join(PROCESSED_DIR, "hard_samples.parquet")

    # Neural Model
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.json")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "rag_transformer_best.pt")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Data Processing
    MAX_SEQ_LEN = 128  # Maximum length for character sequences
    RETRIEVAL_K = 1  # Number of neighbors to retrieve

    # Transformer Architecture
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    N_HEADS = 4
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 4
    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3
    GRADIENT_CLIP_VAL = 1.0

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.STATS_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")
