import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this experiment (Idea 7)
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Paths (referenced by metadata)
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # Text Processing
    MAX_LENGTH = 85  # Based on Mean + 3*Std analysis

    # TF-IDF Parameters
    TFIDF_MIN_DF = 2
    TFIDF_WORD_NGRAM_RANGE = (1, 3)
    TFIDF_CHAR_NGRAM_RANGE = (2, 5)

    # Dimensionality Reduction
    SVD_N_COMPONENTS = 100

    # Stylometric Features
    # List of punctuation characters to count
    PUNCTUATION_CHARS = [";", ":", "!", "?", ",", '"', "-"]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Neural Backbone
    MODEL_NAME = "microsoft/deberta-v3-large"

    # Training Parameters
    # Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS = 8 * 2 = 16
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 2
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Regularization
    # Multi-Sample Dropout rates
    DROPOUT_RATES = [0.1, 0.1, 0.2, 0.2, 0.3]

    # Early Stopping
    PATIENCE = 1  # Aggressive early stopping

    # =========================================================================
    # Caching Filenames (for reproducibility and speed)
    # =========================================================================
    # Feature Caches
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.npy")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    # OOF Predictions Caches
    CACHE_OOF_NN = os.path.join(WORKING_DIR, "oof_nn.npy")
    CACHE_OOF_LR = os.path.join(WORKING_DIR, "oof_lr.npy")
    CACHE_OOF_NB = os.path.join(WORKING_DIR, "oof_nb.npy")
    CACHE_OOF_XGB = os.path.join(WORKING_DIR, "oof_xgb.npy")

    # Test Predictions Caches
    CACHE_PRED_NN = os.path.join(WORKING_DIR, "pred_nn.npy")
    CACHE_PRED_LR = os.path.join(WORKING_DIR, "pred_lr.npy")
    CACHE_PRED_NB = os.path.join(WORKING_DIR, "pred_nb.npy")
    CACHE_PRED_XGB = os.path.join(WORKING_DIR, "pred_xgb.npy")

    # Label Encoder
    CACHE_LABEL_ENCODER = os.path.join(WORKING_DIR, "label_encoder.npy")
