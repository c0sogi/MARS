import os
import shutil
import numpy as np
import pandas as pd
import joblib
import warnings
import logging

# Import from the provided library
from library.config import Config, set_seed
from library.utils import setup_logging, get_logger
from library.feature_engineering import FeaturePipeline
from library.training_pipeline import CVEnsembleTrainer
from library.inference_pipeline import BaggingInference
from library.model_definitions import get_base_learners


def create_demo_data(n_samples=50):
    """
    Creates a small subset of the metadata for demonstration purposes
    to ensure the script runs quickly.
    """
    print(f"Creating demo dataset with {n_samples} samples...")

    # Define demo paths
    demo_meta_dir = "./working/demo_metadata"
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_parquet(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_parquet(Config.VAL_METADATA_PATH)
    test_df = pd.read_parquet(Config.TEST_METADATA_PATH)

    # Sample data
    train_sample = train_df.head(n_samples).copy()
    val_sample = val_df.head(n_samples).copy()
    test_sample = test_df.head(n_samples).copy()

    # Save to demo location
    demo_train_path = os.path.join(demo_meta_dir, "train.parquet")
    demo_val_path = os.path.join(demo_meta_dir, "val.parquet")
    demo_test_path = os.path.join(demo_meta_dir, "test.parquet")

    train_sample.to_parquet(demo_train_path, index=False)
    val_sample.to_parquet(demo_val_path, index=False)
    test_sample.to_parquet(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path


def configure_demo_settings(train_path, val_path, test_path):
    """
    Overrides Config parameters for a fast demonstration run.
    """
    print("Overriding Config parameters for speed...")

    # 1. Update Paths
    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Reduce CV Folds
    Config.N_FOLDS = 2

    # 3. Reduce Model Complexity (Estimators, Depth, etc.)
    # Lexical RF
    Config.LEXICAL_RF_PARAMS["n_estimators"] = 5
    Config.LEXICAL_RF_PARAMS["max_depth"] = 5
    Config.LEXICAL_RF_PARAMS["n_jobs"] = 1  # Avoid overhead in demo

    # Community RF
    Config.COMMUNITY_RF_PARAMS["n_estimators"] = 5
    Config.COMMUNITY_RF_PARAMS["max_depth"] = 5
    Config.COMMUNITY_RF_PARAMS["n_jobs"] = 1
    Config.COMMUNITY_MAX_FEATURES = 50  # Reduce vocab size

    # Semantic XGB
    Config.SEMANTIC_XGB_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_XGB_PARAMS["max_depth"] = 3
    Config.SEMANTIC_XGB_PARAMS["device"] = (
        "cpu"  # Use CPU for tiny data to avoid CUDA init overhead
    )

    # Semantic RF
    Config.SEMANTIC_RF_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_RF_PARAMS["max_depth"] = 5
    Config.SEMANTIC_RF_PARAMS["n_jobs"] = 1

    # Temporal LGBM
    Config.TEMPORAL_LGBM_PARAMS["n_estimators"] = 5
    Config.TEMPORAL_LGBM_PARAMS["num_leaves"] = 5
    Config.TEMPORAL_LGBM_PARAMS["n_jobs"] = 1

    # Metadata LogReg
    Config.METADATA_LOGREG_PARAMS["max_iter"] = 50
    Config.METADATA_LOGREG_PARAMS["n_jobs"] = 1


def run_demo():
    # Setup
    set_seed(42)
    setup_logging(level=logging.INFO)
    logger = get_logger("DemoScript")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Step 1: Prepare Data & Config
    demo_train, demo_val, demo_test = create_demo_data(
        n_samples=40
    )  # Small N for speed
    configure_demo_settings(demo_train, demo_val, demo_test)

    # Step 2: Feature Engineering
    logger.info(">>> STEP 2: Feature Engineering Demo")
    pipeline = FeaturePipeline()

    # Force re-computation by ignoring cache (though cache dir is new/empty)
    X_train_dict, y_train, X_test_dict, test_ids = pipeline.get_data(
        load_cached_data=False
    )

    # Validation
    assert (
        len(y_train) == 80
    ), f"Expected 80 training samples (40 train + 40 val), got {len(y_train)}"
    assert len(test_ids) == 40, f"Expected 40 test samples, got {len(test_ids)}"
    assert "lexical" in X_train_dict
    assert "semantic" in X_train_dict
    assert X_train_dict["semantic"].shape[1] == 384, "BERT embeddings should be 384 dim"
    logger.info("Feature Engineering output shapes verified.")

    # Step 3: Model Definitions Check
    logger.info(">>> STEP 3: Model Definitions Check")
    factories = get_base_learners()
    assert "lexical_bagger" in factories
    rf_model = factories["lexical_bagger"]()
    assert rf_model.n_estimators == 5, "Config override failed for RF"
    logger.info("Model factories loaded and configured correctly.")

    # Step 4: Training Pipeline
    logger.info(">>> STEP 4: Training Pipeline Demo")
    trainer = CVEnsembleTrainer()
    trainer.run()

    # Validation
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    expected_models_count = len(factories) * Config.N_FOLDS + 1  # +1 for meta learner
    files = os.listdir(models_dir)
    joblib_files = [f for f in files if f.endswith(".joblib")]

    assert (
        len(joblib_files) == expected_models_count
    ), f"Expected {expected_models_count} model files, found {len(joblib_files)}"

    oof_path = os.path.join(
        Config.WORKING_DIR, "predictions", "oof_predictions.parquet"
    )
    assert os.path.exists(oof_path), "OOF predictions file missing"
    logger.info("Training complete. Models and OOF predictions verified.")

    # Step 5: Inference Pipeline
    logger.info(">>> STEP 5: Inference Pipeline Demo")
    inference = BaggingInference()
    inference.predict(load_cached_data=True)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file missing"
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert submission_df.shape == (
        40,
        2,
    ), f"Submission shape mismatch: {submission_df.shape}"
    assert Config.ID_COL in submission_df.columns
    assert Config.TARGET_COL in submission_df.columns

    # Check values are probabilities
    preds = submission_df[Config.TARGET_COL]
    assert (
        preds.min() >= 0 and preds.max() <= 1
    ), "Predictions out of probability range [0, 1]"

    logger.info("Inference complete. Submission file verified.")
    logger.info(">>> DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_demo()
