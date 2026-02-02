import os
import torch


class Config:
    """
    Centralized configuration for the Chatbot Arena Prediction task.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data Paths
    # ==========================================
    # Input Metadata (Generated previously)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output/Working Directories
    WORKING_DIR = "./working/idea_1"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Cache File Paths (for numpy arrays)
    # These store the pre-computed embeddings and labels to speed up experiments
    TRAIN_EMBEDS_PATH = os.path.join(CACHE_DIR, "train_embeds.npy")
    TRAIN_LABELS_PATH = os.path.join(CACHE_DIR, "train_labels.npy")

    VAL_EMBEDS_PATH = os.path.join(CACHE_DIR, "val_embeds.npy")
    VAL_LABELS_PATH = os.path.join(CACHE_DIR, "val_labels.npy")

    TEST_EMBEDS_PATH = os.path.join(CACHE_DIR, "test_embeds.npy")
    TEST_IDS_PATH = os.path.join(CACHE_DIR, "test_ids.npy")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "mlp_model.pth")

    # ==========================================
    # Model Architecture
    # ==========================================
    # Sentence Transformer for feature extraction
    SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Embedding Dimensions
    # all-MiniLM-L6-v2 outputs 384-dim vectors.
    # We concatenate Prompt (384) + Response A (384) + Response B (384)
    # Plus 7 scalar features (len_p, len_a, len_b, len_diff, len_ratio, nl_a, nl_b)
    INPUT_DIM = 384 * 3 + 7

    # MLP Architecture
    HIDDEN_DIM = 512
    OUTPUT_DIM = 3  # Classes: model_a, model_b, tie
    DROPOUT_RATE = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3
