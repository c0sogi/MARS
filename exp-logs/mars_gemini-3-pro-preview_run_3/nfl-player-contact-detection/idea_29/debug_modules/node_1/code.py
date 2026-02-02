import pandas as pd
import numpy as np
import os
import sys
import shutil

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.dataset_builder import DatasetBuilder
from library.model_trainer import (
    train_and_validate,
    generate_submission,
    DualStreamPredictor,
)
from library.evaluator import Evaluator


def main():
    # 1. Setup and Reproducibility
    seed_everything(42)
    print("Starting demonstration script...")

    # Define paths for temporary mini-datasets
    working_dir = "./working/demo_run"
    os.makedirs(working_dir, exist_ok=True)

    mini_train_path = os.path.join(working_dir, "mini_train.csv")
    mini_val_path = os.path.join(working_dir, "mini_validation.csv")
    mini_test_path = os.path.join(working_dir, "mini_test.csv")
    mini_sub_path = os.path.join(working_dir, "mini_sample_submission.csv")
    submission_output_path = os.path.join(working_dir, "submission", "submission.csv")

    # 2. Create Mini Datasets (Subsetting)
    print("Creating mini-datasets for rapid execution...")

    # Load original metadata
    # We assume these files exist as per the prompt description
    orig_train = pd.read_csv(Config.TRAIN_META_PATH)
    orig_val = pd.read_csv(Config.VAL_META_PATH)
    orig_test = pd.read_csv(Config.TEST_META_PATH)

    # Select a small number of plays to ensure pipeline runs quickly but has data
    # 5 plays for train to ensure we get some positive contacts
    train_plays = orig_train["game_play"].unique()[:5]
    val_plays = orig_val["game_play"].unique()[:2]
    test_plays = orig_test["game_play"].unique()[:2]

    df_mini_train = orig_train[orig_train["game_play"].isin(train_plays)].copy()
    df_mini_val = orig_val[orig_val["game_play"].isin(val_plays)].copy()
    df_mini_test = orig_test[orig_test["game_play"].isin(test_plays)].copy()

    # Save mini datasets to working directory
    df_mini_train.to_csv(mini_train_path, index=False)
    df_mini_val.to_csv(mini_val_path, index=False)
    df_mini_test.to_csv(mini_test_path, index=False)

    # Create a matching sample_submission for the test subset
    df_mini_sub = df_mini_test[["contact_id"]].copy()
    df_mini_sub["contact"] = 0
    df_mini_sub.to_csv(mini_sub_path, index=False)

    print(
        f"Mini-datasets created: Train={len(df_mini_train)}, Val={len(df_mini_val)}, Test={len(df_mini_test)}"
    )

    # 3. Runtime Configuration Override
    print("Overriding Config parameters...")

    # Redirect paths to our mini datasets
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path
    Config.TEST_META_PATH = mini_test_path
    Config.SAMPLE_SUBMISSION_PATH = mini_sub_path
    Config.WORKING_DIR = working_dir
    Config.SUBMISSION_PATH = submission_output_path

    # Reduce Model Complexity for Speed
    # We use a very small number of estimators (trees)
    Config.STREAM_A_MODEL_PARAMS["n_estimators"] = 2
    Config.STREAM_B_MODEL_PARAMS["n_estimators"] = 2

    # Disable verbose logging from XGBoost
    Config.VERBOSE_EVAL = False

    # Simplify Threshold Search
    Config.THRESHOLD_SEARCH = (0.4, 0.6, 0.1)

    # 4. Demonstrate Feature Engineering (DatasetBuilder)
    print("\n--- Demonstrating DatasetBuilder & FeatureEngineer ---")

    # Initialize builder for 'train' split
    # load_cached_data=False forces generation from our new mini-csvs
    builder = DatasetBuilder("train", load_cached_data=False)

    # Build Stream A (Player-Player Interaction)
    print("Building Stream A features...")
    X_a, ids_a, y_a = builder.build_dataset("A")

    # Verify Stream A
    assert not X_a.empty, "Stream A feature DataFrame is empty."
    assert len(X_a) == len(y_a), "Stream A features and labels length mismatch."
    assert "distance" in X_a.columns, "Stream A missing 'distance' feature."
    assert "sideline_iou" in X_a.columns, "Stream A missing 'sideline_iou' feature."
    print(f"Stream A built successfully. Shape: {X_a.shape}")

    # Build Stream B (Player-Ground Impact)
    print("Building Stream B features...")
    X_b, ids_b, y_b = builder.build_dataset("B")

    # Verify Stream B
    # Note: Small subsets might occasionally lack ground contacts, but with 5 plays it's unlikely.
    if not X_b.empty:
        assert len(X_b) == len(y_b), "Stream B features and labels length mismatch."
        assert "v_surge" in X_b.columns, "Stream B missing 'v_surge' feature."
        print(f"Stream B built successfully. Shape: {X_b.shape}")
    else:
        print("Stream B is empty (no ground contacts in subset).")

    # 5. Demonstrate Model Training (ModelTrainer)
    print("\n--- Demonstrating Model Training Pipeline ---")

    # train_and_validate orchestrates the whole process:
    # loading data, undersampling, training XGBoost, and optimizing thresholds.
    # We set load_cached_data=True so it picks up the parquet files we just generated in step 4.
    predictor = train_and_validate(load_cached_data=True)

    # Verify Predictor
    assert isinstance(
        predictor, DualStreamPredictor
    ), "Trainer failed to return a DualStreamPredictor instance."
    assert hasattr(predictor.model_a, "predict"), "Model A is not a valid model object."
    assert hasattr(predictor.model_b, "predict"), "Model B is not a valid model object."
    print(
        f"Models trained. Optimized Thresholds - A: {predictor.threshold_a:.4f}, B: {predictor.threshold_b:.4f}"
    )

    # 6. Demonstrate Evaluation (Evaluator)
    print("\n--- Demonstrating Evaluator ---")

    # We'll evaluate the trained model on the training set (Stream A) just to show the API usage.
    # In a real scenario, this would be done on the validation set.
    probs_a = predictor.predict_proba(X_a, "A")

    # Optimize threshold manually using Evaluator
    best_thresh = Evaluator.optimize_threshold(y_a, probs_a, stream_name="A_Demo")

    # Calculate MCC at that threshold
    mcc = Evaluator.evaluate(y_a, probs_a, best_thresh, stream_name="A_Demo")

    # Sanity check on metric
    assert -1.0 <= mcc <= 1.0, "MCC score is out of valid range [-1, 1]."
    print(f"Evaluator demonstration complete. MCC: {mcc:.4f}")

    # 7. Demonstrate Submission Generation
    print("\n--- Demonstrating Submission Generation ---")

    # This generates predictions for the test set (our mini_test.csv)
    generate_submission(predictor, load_cached_data=False)

    # Verify Output
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(submission_df) == len(
        df_mini_sub
    ), f"Submission length mismatch. Expected {len(df_mini_sub)}, got {len(submission_df)}"
    assert (
        "contact_id" in submission_df.columns and "contact" in submission_df.columns
    ), "Submission columns are incorrect."

    print("Submission generated and verified successfully.")
    print("\nAll demonstrations completed.")


if __name__ == "__main__":
    main()
