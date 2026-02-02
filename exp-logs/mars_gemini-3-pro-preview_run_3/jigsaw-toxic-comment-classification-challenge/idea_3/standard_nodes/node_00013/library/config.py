import os
import torch


class Config:
    # --- General Configuration ---
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Data Paths ---
    # Raw Data
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Splits)
    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --- Labels ---
    LABEL_COLS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    NUM_LABELS = 6

    # --- DeBERTa Model Hyperparameters ---
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LEN = 512

    # Training
    EPOCHS = 2
    TRAIN_BATCH_SIZE = 8  # Tuned for A100 40GB with Large model @ 512 seq len
    VALID_BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    LLRD_DECAY = 0.9  # Layer-wise Learning Rate Decay
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.1
    SCHEDULER = "cosine"

    # --- NBSVM Hyperparameters ---
    NBSVM_WORD_NGRAM_RANGE = (1, 2)
    NBSVM_CHAR_NGRAM_RANGE = (2, 6)
    NBSVM_MIN_DF = 3
    NBSVM_C = 1.0

    # --- Caching Paths (Intermediate Artifacts) ---
    # NBSVM Features (Sparse Matrices)
    CACHE_NBSVM_WORD_TRAIN = os.path.join(WORKING_DIR, "nbsvm_word_train.npz")
    CACHE_NBSVM_WORD_TEST = os.path.join(WORKING_DIR, "nbsvm_word_test.npz")
    CACHE_NBSVM_CHAR_TRAIN = os.path.join(WORKING_DIR, "nbsvm_char_train.npz")
    CACHE_NBSVM_CHAR_TEST = os.path.join(WORKING_DIR, "nbsvm_char_test.npz")
    CACHE_NBSVM_LABELS = os.path.join(WORKING_DIR, "nbsvm_labels.npy")

    # DeBERTa Checkpoint
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "deberta_best.pth")

    # Predictions (for Ensembling)
    PRED_DEBERTA_VAL = os.path.join(WORKING_DIR, "pred_deberta_val.npy")
    PRED_DEBERTA_TEST = os.path.join(WORKING_DIR, "pred_deberta_test.npy")
    PRED_NBSVM_VAL = os.path.join(WORKING_DIR, "pred_nbsvm_val.npy")
    PRED_NBSVM_TEST = os.path.join(WORKING_DIR, "pred_nbsvm_test.npy")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
