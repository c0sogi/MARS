import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 15 (FPN-Enhanced Deformable Siamese)
    WORK_DIR = "./working/idea_15"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Parquet/Numpy)
    CACHE_TRAIN_PATH = os.path.join(WORK_DIR, "processed_train.parquet")
    CACHE_VAL_PATH = os.path.join(WORK_DIR, "processed_val.parquet")
    CACHE_TEST_PATH = os.path.join(WORK_DIR, "processed_test.parquet")
    CACHE_AGE_STATS = os.path.join(WORK_DIR, "age_stats.npy")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Input: Image (1) + Age (1) + Implant (1) = 3 Channels
    IN_CHANNELS = 3
    IMG_SIZE = (768, 768)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "tf_efficientnet_b2_ns"
    DROP_RATE = 0.2
    DROP_PATH_RATE = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 10
    # Batch size: EfficientNet-B2 @ 768x768 is VRAM intensive.
    # Siamese network doubles the memory footprint per sample.
    BATCH_SIZE = 8
    VAL_BATCH_SIZE = 16

    # Optimizer
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function
    # Aggressive positive weighting for 1:47 imbalance
    POS_WEIGHT = 47.0

    # Gradient Handling
    # Explicitly disabled to allow large updates for minority class
    MAX_GRAD_NORM = None

    # Scheduler (Cosine Annealing)
    T_MAX = 10
    ETA_MIN = 1e-6

    # =========================================================================
    # Debugging & Control
    # =========================================================================
    DEBUG = False

    @classmethod
    def update(cls, debug=False, epochs=None, batch_size=None):
        """
        Updates configuration based on runtime arguments.
        """
        if debug:
            cls.DEBUG = True
            cls.EPOCHS = 2
            cls.BATCH_SIZE = 4
            cls.NUM_WORKERS = 0  # Easier debugging on main thread
            print(
                f"[Config] Debug mode enabled. Epochs={cls.EPOCHS}, Batch={cls.BATCH_SIZE}"
            )

        if epochs is not None:
            cls.EPOCHS = epochs
            cls.T_MAX = epochs

        if batch_size is not None:
            cls.BATCH_SIZE = batch_size
