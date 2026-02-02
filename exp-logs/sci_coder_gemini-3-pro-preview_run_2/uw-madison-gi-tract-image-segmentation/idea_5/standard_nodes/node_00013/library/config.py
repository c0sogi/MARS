import os


class Config:
    """
    Configuration for Idea 5: Retrieval-based Multi-Atlas Segmentation.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and models
    WORKING_DIR = "./working/idea_5"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache File Paths (for the Atlas Bank)
    # Stores the metadata (depth, case, etc.) for the search index
    ATLAS_INDEX_PATH = os.path.join(WORKING_DIR, "atlas_index.parquet")
    # Stores the downsampled image vectors for similarity search
    ATLAS_VECTORS_PATH = os.path.join(WORKING_DIR, "atlas_vectors.npy")
    # Stores the resized binary masks for the atlas
    ATLAS_MASKS_PATH = os.path.join(WORKING_DIR, "atlas_masks.npy")

    # =========================================================================
    # Image Processing Parameters
    # =========================================================================
    # Target size for the final segmentation fusion (Height, Width)
    # Images are resized to this for pixel-wise consensus
    IMG_SIZE = (256, 256)

    # Target size for the retrieval vector (Height, Width)
    # Images are downsampled to this to create a lightweight search vector
    SEARCH_SIZE = (32, 32)

    # =========================================================================
    # Model / Algorithm Hyperparameters
    # =========================================================================
    # Number of similar atlas slices to retrieve for each test slice
    K_NEIGHBORS = 15

    # Relative depth tolerance for search window (0.0 to 1.0)
    # e.g., 0.05 means we search within +/- 5% of the test slice's relative depth
    DEPTH_TOLERANCE = 0.05

    # Threshold for majority voting (0.0 to 1.0)
    # A pixel is predicted as positive if >= FUSION_THRESHOLD of retrieved masks agree
    FUSION_THRESHOLD = 0.5

    # =========================================================================
    # Data & Reproducibility
    # =========================================================================
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = len(CLASSES)

    SEED = 42

    # Number of CPU workers for data loading (if applicable)
    NUM_WORKERS = 4

    # Debug flag to run on a smaller subset of data
    DEBUG = False
