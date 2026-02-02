import os


class Config:
    # --------------------------------------------------------------------------
    # Global Configuration & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Ensure working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Processing & Volumetric MIP Hyperparameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 224

    # Volumetric MIP Stacking Logic
    # We select 3 slabs (Upper, Center, Lower) centered around the FLAIR max intensity slice.
    NUM_SLABS = 3
    SLAB_SIZE = 1  # Cite solution_lesson_node_00029: Use single slices, not MIP
    SLAB_STRIDE = 5  # Distance between the centers of the slabs
    ROI_EXCLUDE_BUFFER = 0.15  # Cite solution_lesson_node_00018: Exclude boundaries

    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

    # Total input channels = Modalities * Slabs (4 * 3 = 12 channels)
    IN_CHANNELS = NUM_MODALITIES * NUM_SLABS

    # Caching Paths (using .npy for numerical arrays as per instructions)
    # These will be used by the Dataset class to store/load processed tensors
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_cache.npy")

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.3  # Increased regularization to combat overfitting

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32  # Conservative batch size for 12-channel input on A100
    NUM_EPOCHS = 20  # Sufficient for convergence on small dataset
    LEARNING_RATE = 1e-4  # Standard starting LR for fine-tuning
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay to prevent overfitting
    PATIENCE = 5  # Early stopping patience

    # Augmentation
    ROTATION_DEGREES = 15  # Strictly limit rotation to +/- 15 degrees
