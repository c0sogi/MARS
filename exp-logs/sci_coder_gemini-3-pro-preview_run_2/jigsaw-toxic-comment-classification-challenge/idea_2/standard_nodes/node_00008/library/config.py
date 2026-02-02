import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Dataset Paths (using stratified metadata splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_CSV = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (for Branch A features)
    # We use .npz or folder structures for sparse matrices to avoid pickle
    CACHE_DIR = WORKING_DIR

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    LABEL_COLS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    ID_COL = "id"
    TEXT_COL = "comment_text"

    # --------------------------------------------------------------------------
    # General System Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging / Development
    # Set DEBUG = True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000

    # --------------------------------------------------------------------------
    # Branch A: Linear Ensemble (TF-IDF + Logistic Regression)
    # --------------------------------------------------------------------------
    # Feature Extraction
    TFIDF_WORD_MAX_FEATURES = 20000
    TFIDF_WORD_NGRAM_RANGE = (1, 2)

    TFIDF_CHAR_MAX_FEATURES = 30000
    TFIDF_CHAR_NGRAM_RANGE = (2, 6)

    # Model Hyperparameters
    # We use multiple C values to create a diverse ensemble of linear models
    LR_C_VALUES = [0.1, 0.5, 1.0, 2.0, 4.0]
    LR_SOLVER = "sag"
    LR_MAX_ITER = 1000
    LR_N_JOBS = 12

    # --------------------------------------------------------------------------
    # Branch B: Deep Learning (Transformer)
    # --------------------------------------------------------------------------
    # Model Architecture
    # We use multiple architectures for diversity (Cite solution_lesson_node_00007)
    MODEL_NAMES = ["roberta-base", "bert-base-uncased"]

    # Tokenization
    MAX_LEN = 128  # Covers the majority of comment lengths efficiently

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Optimization
    PATIENCE = 1  # Early stopping patience

    # --------------------------------------------------------------------------
    # Ensemble Strategy
    # --------------------------------------------------------------------------
    # Weighted averaging of probabilities
    # Transformer gets higher weight due to semantic capabilities,
    # but Linear provides robustness against OOV/noise.
    ENSEMBLE_WEIGHTS = {"linear": 0.3, "transformer": 0.7}

    @staticmethod
    def setup():
        """
        Creates the necessary working and submission directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
