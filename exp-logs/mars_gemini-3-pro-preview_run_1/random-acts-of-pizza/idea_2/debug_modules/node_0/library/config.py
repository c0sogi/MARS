import os


class Config:
    """
    Configuration class for the Hybrid Semantic-Tabular Random Forest project.
    Defines file paths, model hyperparameters, and execution settings.
    """

    # global random seed for reproducibility
    RANDOM_SEED = 42

    # -------------------------------------------------------------------------
    # Directory Configuration
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist (though creation is usually handled by processing scripts)
    # We define paths here for consistency across modules.

    # -------------------------------------------------------------------------
    # Input Data Paths (Generated Metadata)
    # -------------------------------------------------------------------------
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Caching Paths (Processed Features)
    # -------------------------------------------------------------------------
    # Parquet files to store the fused semantic + tabular features
    TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # -------------------------------------------------------------------------
    # Submission Path
    # -------------------------------------------------------------------------
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------

    # Text Embedding
    # Using a lightweight but effective sentence transformer
    TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Random Forest Classifier
    # Parameters tuned for robustness on small, high-dimensional data
    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH = 25
    RF_MAX_FEATURES = "sqrt"
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set to a positive integer (e.g., 100) to limit dataset size for quick debugging.
    # Set to None to use the full dataset.
    DEBUG_SAMPLE_SIZE = None
