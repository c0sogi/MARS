import os
import torch

# Ensure the working directory exists immediately upon import
WORKING_DIR = "./working/idea_8"
os.makedirs(WORKING_DIR, exist_ok=True)


class Config:
    """
    Project-wide constants and hyperparameters for the High-Capacity
    Smoothed Semantic Regressor (HC-SSR) solution.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = WORKING_DIR

    # Metadata Source Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Artifacts
    # Directory to save the fine-tuned MPNet model
    BACKBONE_OUTPUT_DIR = os.path.join(WORKING_DIR, "fine_tuned_mpnet")
    # File to save the trained LightGBM model
    LGBM_MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")
    # Final submission file
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Feature Cache Files (Parquet format preferred over pickle)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # =========================================================================
    # Model Hyperparameters (Semantic Backbone)
    # =========================================================================
    # High-capacity backbone to resolve abstract markdown descriptions
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Max sequence length for tokenization
    MAX_LENGTH = 128

    # Batch size for training and inference
    BATCH_SIZE = 32

    # Fine-tuning parameters for the Contrastive Learning stage
    NUM_EPOCHS = 1
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 0.01

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================
    # 1D Convolution Kernel for smoothing similarity vectors.
    # Used to identify the "dense center" of semantic matches.
    SMOOTHING_KERNEL = [0.2, 0.6, 0.2]

    # =========================================================================
    # Regressor Hyperparameters (LightGBM)
    # =========================================================================
    # Parameters for the ranking regression stage
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": 12,  # Utilize available vCPUs
        "verbose": -1,  # Silent mode
        "random_state": 42,
    }

    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging flags
    # Set DEBUG to True to run on a smaller subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000
