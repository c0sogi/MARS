import os
import sys
import pandas as pd
import numpy as np
import library.config as config
from library.data_processor import TaxiDataProcessor
from library.model_trainer import XGBTrainer
from library.evaluator import ModelEvaluator


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    print("=== Starting Library Usage Demonstration ===")

    # ---------------------------------------------------------
    # 1. Prepare Mini Dataset for Speed
    # ---------------------------------------------------------
    # We create a small subset of the data to ensure the demo runs in seconds/minutes
    # rather than training on the full 55M row dataset.
    print("\n[Step 1] Creating mini-datasets for rapid testing...")

    demo_dir = "./working/demo_data"
    os.makedirs(demo_dir, exist_ok=True)

    mini_train_path = os.path.join(demo_dir, "train.parquet")
    mini_val_path = os.path.join(demo_dir, "val.parquet")
    mini_test_path = os.path.join(demo_dir, "test.parquet")

    # Load a small head of the original data
    # Note: We access the original paths via the imported config before we patch it
    print("  Reading subset of original data...")
    orig_train = pd.read_parquet(config.PATH_CONFIG["train_data"]).head(5000)
    orig_val = pd.read_parquet(config.PATH_CONFIG["val_data"]).head(1000)
    orig_test = pd.read_parquet(config.PATH_CONFIG["test_data"]).head(500)

    # Save to the working directory
    orig_train.to_parquet(mini_train_path, index=False)
    orig_val.to_parquet(mini_val_path, index=False)
    orig_test.to_parquet(mini_test_path, index=False)

    print(f"  Mini-datasets saved to {demo_dir}")

    # ---------------------------------------------------------
    # 2. Configure Environment (Monkey Patching)
    # ---------------------------------------------------------
    print("\n[Step 2] Configuring parameters for demo run...")

    # Override data paths to point to our mini-dataset
    # Since PATH_CONFIG is a mutable dictionary imported by other modules,
    # changing it here updates the reference used by TaxiDataProcessor.
    config.PATH_CONFIG["train_data"] = mini_train_path
    config.PATH_CONFIG["val_data"] = mini_val_path
    config.PATH_CONFIG["test_data"] = mini_test_path

    # Override output paths to avoid overwriting any real work
    config.PATH_CONFIG["model_save_path"] = "./working/demo_outputs/xgb_model.json"
    config.PATH_CONFIG["submission_output"] = "./working/demo_outputs/submission.csv"

    # Reduce training complexity
    config.TRAIN_CONFIG["num_boost_round"] = 10  # Very few rounds for demo
    config.TRAIN_CONFIG["early_stopping_rounds"] = 5
    config.TRAIN_CONFIG["verbose_eval"] = 1

    # Ensure output directory exists
    os.makedirs("./working/demo_outputs", exist_ok=True)

    # ---------------------------------------------------------
    # 3. Demonstrate Data Processor
    # ---------------------------------------------------------
    print("\n[Step 3] Demonstrating TaxiDataProcessor...")
    processor = TaxiDataProcessor()

    # Process train data
    # IMPORTANT: set load_cached_data=False to force reading our new mini-dataset
    # instead of any cached files from previous full runs.
    print("  Processing training data...")
    train_df = processor.get_processed_data("train", load_cached_data=False)

    # Validation Logic
    print("  Validating processed data...")
    expected_cols = ["dist_haversine", "dist_manhattan", "hour", "year", "fare_amount"]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing expected column: {col}"

    assert len(train_df) == 5000, f"Expected 5000 rows, found {len(train_df)}"

    # Check coordinate clamping (sanity check)
    assert (
        train_df["pickup_latitude"].max() <= config.BOUNDING_BOX["lat_max"]
    ), "Latitude clamping failed"
    assert (
        train_df["pickup_latitude"].min() >= config.BOUNDING_BOX["lat_min"]
    ), "Latitude clamping failed"

    print("  TaxiDataProcessor logic verified.")

    # ---------------------------------------------------------
    # 4. Demonstrate Model Trainer
    # ---------------------------------------------------------
    print("\n[Step 4] Demonstrating XGBTrainer...")
    trainer = XGBTrainer()

    # Train the model
    print("  Starting training loop...")
    trainer.train(load_cached_data=False)

    # Validation Logic
    assert os.path.exists(
        config.PATH_CONFIG["model_save_path"]
    ), "Model file was not created."
    print(f"  Model successfully saved to {config.PATH_CONFIG['model_save_path']}")

    # ---------------------------------------------------------
    # 5. Demonstrate Evaluator
    # ---------------------------------------------------------
    print("\n[Step 5] Demonstrating ModelEvaluator...")
    evaluator = ModelEvaluator()

    # Calculate Metrics
    print("  Calculating validation metrics...")
    rmse = evaluator.calculate_metrics(load_cached_data=False)
    print(f"  Resulting RMSE: {rmse:.4f}")

    assert isinstance(rmse, float), "RMSE should be a float"
    assert rmse >= 0, "RMSE cannot be negative"

    # Generate Submission
    print("  Generating submission file...")
    evaluator.generate_submission(load_cached_data=False)

    # Validation Logic
    sub_path = config.PATH_CONFIG["submission_output"]
    assert os.path.exists(sub_path), "Submission file not found"

    sub_df = pd.read_csv(sub_path)
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns are incorrect"
    assert len(sub_df) == 500, f"Expected 500 predictions, found {len(sub_df)}"

    # Check if predictions are valid (e.g., >= 2.50 as per post-processing rule)
    assert sub_df["fare_amount"].min() >= 2.50, "Post-processing min fare check failed"

    print(f"  Submission verified at {sub_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
