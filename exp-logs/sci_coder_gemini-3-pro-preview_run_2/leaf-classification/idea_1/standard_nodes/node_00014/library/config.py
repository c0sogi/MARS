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
    # Implements Regularized Multinomial Logistic Regression with CV-tuned C
    # Updated to use ElasticNet (L1+L2) with SAGA solver to handle feature redundancy
    # Cite solution_lesson_node_00004: Dynamic Regularization Tuning
    MODEL_PARAMS = {
        "Cs": 20,  # Grid of 20 values for C (log-spaced)
        "cv": 5,  # 5-fold Cross Validation
        "scoring": "neg_log_loss",  # Optimize for Log Loss
        "penalty": "elasticnet",
        "solver": "saga",
        "l1_ratios": [0.0, 0.1, 0.5, 0.7, 0.9, 0.95, 1.0],  # Mix of L1 and L2
        "multi_class": "multinomial",
        "max_iter": 5000,  # Increased iterations for SAGA convergence
        "tol": 1e-4,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # ==========================================
    # Preprocessing Settings
    # ==========================================
    SCALE_FEATURES = True  # Apply StandardScaling (Z-score) to features
