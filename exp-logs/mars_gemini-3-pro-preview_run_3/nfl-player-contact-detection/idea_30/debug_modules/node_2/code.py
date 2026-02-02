import os
import pandas as pd
import numpy as np
import xgboost as xgb
import sys

# Import provided library modules
from library.config import ProjectConfig
from library.utils import seed_everything, get_logger
from library.data_pipeline import DataPipeline
from library.model_trainer import DualStreamTrainer
from library.metric_optimizer import MetricOptimizer


def run_demo():
    # 1. Setup and Configuration
    logger = get_logger("Demo")
    logger.info("Starting End-to-End Demo...")

    # Set seed for reproducibility
    seed_everything(42)

    # Define working directories for the demo
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Create Mini-Datasets to speed up execution
    # We will sample a small number of plays from the original metadata
    # and save them as temporary metadata files. The DataPipeline will
    # then filter the large raw CSVs down to just these plays.

    logger.info("Creating mini-datasets for rapid demonstration...")

    def create_mini_metadata(source_path, dest_path, n_plays=2):
        df = pd.read_csv(source_path)
        # Select first n unique plays
        plays = df["game_play"].unique()[:n_plays]
        df_mini = df[df["game_play"].isin(plays)].copy()
        df_mini.to_csv(dest_path, index=False)
        return df_mini

    # Paths for mini metadata
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    # Generate mini files
    # Using 2 plays for train, 1 for val, 1 for test
    df_mini_train = create_mini_metadata(
        ProjectConfig.TRAIN_META_PATH, mini_train_path, n_plays=2
    )
    df_mini_val = create_mini_metadata(
        ProjectConfig.VAL_META_PATH, mini_val_path, n_plays=1
    )
    df_mini_test = create_mini_metadata(
        ProjectConfig.TEST_META_PATH, mini_test_path, n_plays=1
    )

    logger.info(f"Mini Train Shape: {df_mini_train.shape}")
    logger.info(f"Mini Val Shape: {df_mini_val.shape}")
    logger.info(f"Mini Test Shape: {df_mini_test.shape}")

    # 3. Override ProjectConfig for the Demo
    # We modify the class attributes directly to affect the pipeline

    # Point to mini metadata
    ProjectConfig.TRAIN_META_PATH = mini_train_path
    ProjectConfig.VAL_META_PATH = mini_val_path
    ProjectConfig.TEST_META_PATH = mini_test_path

    # Update working directory to avoid conflicts
    ProjectConfig.WORKING_DIR = demo_dir
    ProjectConfig.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    ProjectConfig.SUBMISSION_PATH = os.path.join(
        ProjectConfig.SUBMISSION_DIR, "submission.csv"
    )

    # Reduce Model Complexity for Speed
    # Reduce estimators from 5000 to 10
    ProjectConfig.XGB_PARAMS_STREAM_A["n_estimators"] = 10
    ProjectConfig.XGB_PARAMS_STREAM_B["n_estimators"] = 10

    # Use 'hist' (CPU) or 'gpu_hist' depending on environment, but force hist for stability on small data
    # (Though the config defaults to gpu_hist if available, we'll leave it as is or force it if needed)
    # ProjectConfig.XGB_PARAMS_STREAM_A['tree_method'] = 'hist'
    # ProjectConfig.XGB_PARAMS_STREAM_B['tree_method'] = 'hist'

    # 4. Data Pipeline Execution
    logger.info("Initializing Data Pipeline...")
    pipeline = DataPipeline()

    # Process Train Data
    logger.info("Running Pipeline: Train...")
    data_train = pipeline.run(mode="train", load_cached_data=False)

    # Verify Train Data
    assert not data_train["stream_a"]["X"].empty, "Stream A Train Features are empty"
    assert not data_train["stream_b"]["X"].empty, "Stream B Train Features are empty"
    assert len(data_train["stream_a"]["X"]) == len(
        data_train["stream_a"]["y"]
    ), "Stream A X/y mismatch"

    # Process Validation Data
    logger.info("Running Pipeline: Validation...")
    data_val = pipeline.run(mode="validation", load_cached_data=False)

    # Verify Val Data
    assert not data_val["stream_a"]["X"].empty, "Stream A Val Features are empty"

    # Process Test Data
    logger.info("Running Pipeline: Test...")
    data_test = pipeline.run(mode="test", load_cached_data=False)

    # Verify Test Data
    assert not data_test["stream_a"]["X"].empty, "Stream A Test Features are empty"

    # 5. Model Training
    logger.info("Initializing Model Trainer...")
    trainer = DualStreamTrainer()

    # Train Stream A (Interaction)
    logger.info("Training Stream A...")
    model_a = trainer.train_stream(
        X_train=data_train["stream_a"]["X"],
        y_train=data_train["stream_a"]["y"],
        X_val=data_val["stream_a"]["X"],
        y_val=data_val["stream_a"]["y"],
        stream_type="A",
    )
    assert model_a is not None, "Stream A Model failed to train"

    # Train Stream B (Impact)
    logger.info("Training Stream B...")
    model_b = trainer.train_stream(
        X_train=data_train["stream_b"]["X"],
        y_train=data_train["stream_b"]["y"],
        X_val=data_val["stream_b"]["X"],
        y_val=data_val["stream_b"]["y"],
        stream_type="B",
    )
    assert model_b is not None, "Stream B Model failed to train"

    # Save Models
    trainer.save_models(suffix="_demo")
    assert os.path.exists(
        os.path.join(demo_dir, "models", "model_stream_a_demo.json")
    ), "Model A file not saved"

    # 6. Threshold Optimization
    logger.info("Optimizing Thresholds...")
    optimizer = MetricOptimizer()

    # Generate Validation Predictions
    val_probs_a = trainer.predict_stream(model_a, data_val["stream_a"]["X"])
    val_probs_b = trainer.predict_stream(model_b, data_val["stream_b"]["X"])

    thresholds = {}

    # Optimize Stream A
    if len(val_probs_a) > 0:
        thresh_a = optimizer.find_optimal_threshold(
            y_true=data_val["stream_a"]["y"].values,
            y_probs=val_probs_a,
            stream_name="A",
        )
        thresholds["A"] = thresh_a
        assert 0.0 < thresh_a < 1.0, f"Invalid threshold A: {thresh_a}"

    # Optimize Stream B
    if len(val_probs_b) > 0:
        thresh_b = optimizer.find_optimal_threshold(
            y_true=data_val["stream_b"]["y"].values,
            y_probs=val_probs_b,
            stream_name="B",
        )
        thresholds["B"] = thresh_b
        assert 0.0 < thresh_b < 1.0, f"Invalid threshold B: {thresh_b}"

    # 7. Inference and Submission
    logger.info("Generating Submission...")

    # Generate Test Predictions
    test_probs_a = trainer.predict_stream(model_a, data_test["stream_a"]["X"])
    test_probs_b = trainer.predict_stream(model_b, data_test["stream_b"]["X"])

    # Create Submission File
    optimizer.generate_submission(
        probs_a=test_probs_a,
        ids_a=data_test["stream_a"]["ids"],
        probs_b=test_probs_b,
        ids_b=data_test["stream_b"]["ids"],
        thresholds=thresholds,
    )

    # Verify Submission
    sub_path = ProjectConfig.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file was not created"

    df_sub = pd.read_csv(sub_path)
    logger.info(f"Submission generated with {len(df_sub)} rows.")

    expected_cols = ["contact_id", "contact"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {df_sub.columns}"
    assert df_sub["contact"].isin([0, 1]).all(), "Submission contains non-binary values"

    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
