import os
import torch


class Config:
    """
    Configuration class for the Domain-Adapted Heterogeneous Ensemble strategy.
    Handles file paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data for debugging
    DEBUG_SAMPLES = 2000  # Number of samples to use in debug mode

    # Compute Environment
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Artifacts (Cache, Models, TAPT Corpus)
    # Using 'idea_4' as the designated workspace for this iteration
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output Directory for Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    LABEL_COLS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    NUM_LABELS = len(LABEL_COLS)

    # Tokenization
    MAX_LENGTH = 200  # Increased from 128 to 200 to capture longer context

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Branch A: DeBERTa-v3 (To be TAPT-ed then Fine-tuned)
    MODEL_A_NAME = "microsoft/deberta-v3-base"
    # Path to save/load the domain-adapted weights
    MODEL_A_TAPT_PATH = os.path.join(WORKING_DIR, "deberta_v3_tapt_weights")
    MODEL_A_BEST_PATH = os.path.join(WORKING_DIR, "deberta_v3_best_model.bin")

    # Branch B: RoBERTa (To be TAPT-ed then Fine-tuned)
    MODEL_B_NAME = "roberta-base"
    # Path to save/load the domain-adapted weights
    MODEL_B_TAPT_PATH = os.path.join(WORKING_DIR, "roberta_tapt_weights")
    MODEL_B_BEST_PATH = os.path.join(WORKING_DIR, "roberta_best_model.bin")

    # Branch C: Linear Baseline (TF-IDF + Logistic Regression)
    TFIDF_CACHE_DIR = os.path.join(WORKING_DIR, "tfidf_cache")
    os.makedirs(TFIDF_CACHE_DIR, exist_ok=True)
    LINEAR_MODEL_PATH = os.path.join(WORKING_DIR, "linear_ensemble.pkl")

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================

    # Phase 1: Task-Adaptive Pretraining (TAPT)
    # Unsupervised Masked Language Modeling on Train + Test text
    TAPT_PARAMS = {
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "mlm_probability": 0.15,
        "corpus_path": os.path.join(WORKING_DIR, "tapt_corpus.txt"),
        "save_steps": 500,
    }

    # Phase 2: Supervised Fine-Tuning
    # Multi-label classification on labeled data
    TRAIN_PARAMS = {
        "epochs": 3,
        "batch_size": 32,  # A100 40GB can handle this easily
        "lr_backbone": 2e-5,
        "lr_head": 1e-4,  # Higher LR for the classification head
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "llrd_decay": 0.9,  # Layer-wise Learning Rate Decay factor
        "max_grad_norm": 1.0,
        "patience": 2,  # Early stopping patience
        "val_check_interval": 0.5,  # Validate twice per epoch
    }

    # Linear Model Parameters
    LINEAR_PARAMS = {
        "word_ngram_range": (1, 2),
        "char_ngram_range": (2, 6),
        "max_features_word": 20000,
        "max_features_char": 30000,
        "c_val": 1.0,  # Regularization strength for Logistic Regression
        "solver": "sag",
        "n_jobs": -1,
    }

    # Ensemble Optimization
    # Path to store validation predictions for blending optimization
    VAL_PREDS_PATH = os.path.join(WORKING_DIR, "val_predictions.pkl")
    TEST_PREDS_PATH = os.path.join(WORKING_DIR, "test_predictions.pkl")
