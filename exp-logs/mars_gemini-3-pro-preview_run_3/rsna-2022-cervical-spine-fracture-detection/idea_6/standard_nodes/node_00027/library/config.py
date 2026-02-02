import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    DEBUG = False
    SEED = 42
    PROJECT_NAME = "Cervical_Spine_Fracture_Detection"
    IDEA_NAME = "idea_6"  # Anatomically-Conditioned Residual-Instance MIL Network

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs, limit to 12 as per environment specs if needed
    NUM_WORKERS = min(os.cpu_count(), 12)

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATIONS_DIR = os.path.join(INPUT_DIR, "segmentations")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    WORKING_DIR = "./working"
    # Specific cache directory for this idea to store preprocessed .npy files
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    IMAGE_SIZE = (224, 224)
    NUM_SLICES = 64  # Uniform sampling size per exam
    IN_CHANNELS = 3  # 2.5D input: slices z-1, z, z+1

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "resnet18"  # Lighter backbone for effective batch size
    N_CLASSES = 7  # C1 to C7

    # Structural components
    DROPOUT = 0.0  # No dropout in classification head

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8  # Constrained by GPU memory for 3D/MIL tasks
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW
    MAX_GRAD_NORM = 1000

    # Scheduler: Decoupled Cosine Annealing
    T_MAX_MULT = 1.5  # T_max = 1.5 * EPOCHS
    MIN_LR = 1e-6

    # =========================================================================
    # Helper Methods
    # =========================================================================
    @classmethod
    def setup(cls, debug=False):
        """
        Initializes configuration, creates necessary directories, and applies
        overrides if debug mode is enabled.

        Args:
            debug (bool): If True, reduces epochs and dataset size for fast iteration.
        """
        cls.DEBUG = debug

        # Create cache directory if it doesn't exist
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        if cls.DEBUG:
            print(f"[{cls.__name__}] Debug mode enabled.")
            cls.EPOCHS = 2
            cls.BATCH_SIZE = 4
            # We might keep NUM_SLICES same to test memory, or reduce it.
            # Keeping it same ensures model architecture validity.

        print(f"[{cls.__name__}] Configuration Setup:")
        print(f"    Device: {cls.DEVICE}")
        print(f"    Backbone: {cls.BACKBONE}")
        print(f"    Slices per Scan: {cls.NUM_SLICES}")
        print(f"    Batch Size: {cls.BATCH_SIZE}")
        print(f"    Cache Dir: {cls.CACHE_DIR}")
