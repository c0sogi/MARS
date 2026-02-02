import os


class Config:
    """
    Configuration class for the Leaf Classification task.
    Stores file paths, constants, and model hyperparameters.
    """

    # ==========================================
    # Random Seed for Reproducibility
    # ==========================================
    RANDOM_SEED = 42

    # ==========================================
    # Directory Paths
    # ==========================================
    # Root input directory (Read-Only)
    INPUT_DIR = "./input"

    # Metadata directory containing train/val/test splits
    METADATA_DIR = "./metadata"

    # Working directory for caching and intermediate files
    WORKING_DIR = "./working/idea_1"

    # Output directory for final submission
    SUBMISSION_DIR = "./submission"

    # ==========================================
    # File Paths
    # ==========================================
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Sample submission file for format reference
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final output path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache path for processed features (if needed)
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.parquet")

    # ==========================================
    # Data Definitions
    # ==========================================
    ID_COL = "id"
    TARGET_COL = "species"
    IMAGE_PATH_COL = "image_path"

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Parameters for sklearn.linear_model.LogisticRegressionCV
    # Cite solution_lesson_node_00005: Use L2 + L-BFGS for dense features
    # Cite solution_lesson_node_00010: Use cv=3 and dense grid for small datasets
    LR_PARAMS = {
        "Cs": 100,  # Dense grid
        "cv": 3,  # Stabilize tuning on small N
        "scoring": "neg_log_loss",
        "penalty": "l2",
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "max_iter": 5000,
        "tol": 1e-4,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    # Parameters for LinearDiscriminantAnalysis
    # Cite solution_lesson_node_00006: Use LDA with shrinkage
    LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto"}

    # ==========================================
    # Preprocessing Settings
    # ==========================================
    SCALE_FEATURES = True  # Apply StandardScaling (Z-score) to features
