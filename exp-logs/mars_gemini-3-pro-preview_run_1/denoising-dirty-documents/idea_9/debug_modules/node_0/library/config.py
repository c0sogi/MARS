import os
import torch


class Config:
    """
    Configuration for the Heterogeneous Dual-Stream Ensemble for Text Denoising.
    """

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and saving model checkpoints
    # Using idea_9 as the designated workspace
    WORKING_DIR = "./working/idea_9"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # System & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of workers for DataLoader (Adjust based on 12 vCPUs available)
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Training duration: 1000 epochs for full convergence
    NUM_EPOCHS = 1000

    # Optimization
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3

    # Scheduler: Cosine Annealing
    # T_MAX matches NUM_EPOCHS to decay LR to 0 at the end
    T_MAX = 1000

    # Loss Function
    LOSS_FN = "MSE"

    # -------------------------------------------------------------------------
    # Debugging / Development Flags
    # -------------------------------------------------------------------------
    # If set to an integer, limits the number of samples for training/validation.
    # Useful for debugging pipeline without running full dataset.
    # Set to None for full production run.
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    # -------------------------------------------------------------------------
    # Ensemble / Stream Configurations
    # -------------------------------------------------------------------------
    # Stream A: Context Specialists
    # Deep architecture (Depth 4) with large receptive field (Patch 320)
    STREAM_A = {
        "name": "stream_a_context",
        "patch_size": 320,
        "depth": 4,
        "base_channels": 32,
        "num_models": 3,
        # Explicit seeds to ensure diversity among models in this stream
        "seeds": [42, 43, 44],
    }

    # Stream B: Texture Specialists
    # Shallow architecture (Depth 3) with high crop diversity (Patch 160)
    STREAM_B = {
        "name": "stream_b_texture",
        "patch_size": 160,
        "depth": 3,
        "base_channels": 32,
        "num_models": 3,
        # Explicit seeds to ensure diversity among models in this stream
        "seeds": [45, 46, 47],
    }

    # List of streams to iterate over during training
    STREAMS = [STREAM_A, STREAM_B]

    # -------------------------------------------------------------------------
    # Inference / TTA
    # -------------------------------------------------------------------------
    # D4 Group TTA: 8 views (Rotations 0, 90, 180, 270 + Flips)
    TTA_VIEWS = 8

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup upon module import to guarantee directory existence
Config.setup()
