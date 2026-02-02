import os
import torch


class Config:
    """
    Configuration for the Discrete-Topology Wide-Stream Residual BiGRU model.
    Defines global hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_46"

    # Ensure working directory exists for caching and outputs
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Parquet format as generated in metadata step)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Files
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    CACHE_DIR = WORKING_DIR

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEQ_LENGTH = 107
    SCORED_SEQ_LENGTH = 68

    # Targets to be trained on and scored (filtering out high-noise columns)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabularies
    # Atomic Sequence: A, G, C, U
    VOCAB_SIZE_SEQ = 4
    BASE_TO_INT = {"A": 0, "G": 1, "C": 2, "U": 3}

    # Predicted Loop Type: S, M, I, B, H, E, X
    VOCAB_SIZE_LOOP = 7
    LOOP_TO_INT = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Discrete Topological Distance
    # Distances are clipped to [-DIST_CLIP, DIST_CLIP]
    # Then shifted by +DIST_CLIP to be indices [0, 2*DIST_CLIP]
    # We increase clip to 128 to cover full sequence length (107) and use Sinusoidal Encodings
    # Cite solution_lesson_node_00114
    DIST_CLIP = 128
    VOCAB_SIZE_DIST = (DIST_CLIP * 2) + 1

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Embedding Dimensions (Heterogeneous Feature Embedding)
    EMBED_DIM_SEQ = 128
    EMBED_DIM_LOOP = 64
    EMBED_DIM_DIST = 64

    # Total input dimension after concatenation
    INPUT_DIM = EMBED_DIM_SEQ + EMBED_DIM_LOOP + EMBED_DIM_DIST  # 256

    # Recurrent Backbone (Wide-Stream Residual)
    HIDDEN_DIM = 384  # Stream width (192 per direction for BiGRU)
    NUM_LAYERS = 6  # Number of residual blocks (Shallow and Wide)
    DROPOUT = 0.2  # Inter-layer dropout (No Stem dropout)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signal
    GRAD_CLIP = 1.0  # Gradient clipping norm for stability

    # =========================================================================
    # Runtime
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # Debugging flags
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
