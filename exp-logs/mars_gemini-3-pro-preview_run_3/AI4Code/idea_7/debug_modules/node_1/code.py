import os
import shutil
import pandas as pd
import numpy as np
import warnings
import torch

# Import from the provided library
from library.config import Config
from library.data_utils import get_data_splits, set_seed, read_notebook
from library.fine_tuning import run_fine_tuning
from library.feature_engineering import generate_features
from library.lgbm_model import train_regressor, validate_model, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Debugging
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for fast demonstration...")

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.setup()  # Create directories

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use 20 notebooks for extremely fast execution

    # Override Fine-Tuning Hyperparameters
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 2
    Config.EVAL_BATCH_SIZE = 4
    # We use a tiny model for the demo if possible, but the code uses a fixed checkpoint.
    # We will stick to the config's checkpoint but run very few steps.
    # Note: The provided library code uses Config.MODEL_CHECKPOINT directly.
    # To speed things up further, we can't easily change the model architecture
    # without changing the file content, so we rely on small data size.

    # Update Paths based on new WORKING_DIR
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.FINE_TUNED_MODEL_PATH = os.path.join(Config.WORKING_DIR, "dsapr_model")
    Config.LGBM_MODEL_PATH = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Override LightGBM Hyperparameters for speed
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["learning_rate"] = 0.1

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading...")
    df_train, df_val, df_test = get_data_splits()

    print(f"Loaded Train Samples: {len(df_train)}")
    print(f"Loaded Val Samples:   {len(df_val)}")
    print(f"Loaded Test Samples:  {len(df_test)}")

    assert (
        len(df_train) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train split size exceeds debug limit"
    assert len(df_val) <= Config.DEBUG_SAMPLE_SIZE, "Val split size exceeds debug limit"
    assert not df_train.empty, "Train dataframe is empty"

    # -------------------------------------------------------------------------
    # 3. Fine-Tuning Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Semantic Backbone Fine-Tuning...")
    # This will generate pairs from the small train set and fine-tune SBERT
    run_fine_tuning()

    # Verify model artifact creation
    if os.path.exists(Config.FINE_TUNED_MODEL_PATH):
        print("Fine-tuning successful. Model artifact found.")
    else:
        # Note: run_fine_tuning saves to Config.FINE_TUNED_MODEL_PATH (which is a directory in sentence-transformers usually,
        # but the config defines it. Let's check logic).
        # In `library/fine_tuning.py`: output_path=Config.FINE_TUNED_MODEL_PATH
        # If it's a directory, we check existence.
        if os.path.isdir(Config.FINE_TUNED_MODEL_PATH) or os.path.isfile(
            Config.FINE_TUNED_MODEL_PATH
        ):
            print("Fine-tuning successful. Model artifact found.")
        else:
            raise AssertionError(
                f"Fine-tuned model not found at {Config.FINE_TUNED_MODEL_PATH}"
            )

    # -------------------------------------------------------------------------
    # 4. Feature Engineering Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating Features...")

    # Generate features for all splits
    # We force regeneration to ensure it uses the fine-tuned model we just created (or base if failed)
    # and the sampled data.

    print("Generating Train Features...")
    train_features = generate_features(df_train, mode="train", load_cached_data=False)

    print("Generating Validation Features...")
    val_features = generate_features(df_val, mode="val", load_cached_data=False)

    print("Generating Test Features...")
    test_features = generate_features(df_test, mode="test", load_cached_data=False)

    # Verification
    expected_cols = ["best_match_loc", "center_of_mass", "sim_max", "n_code", "md_len"]

    # Check Train
    if not train_features.empty:
        assert all(
            col in train_features.columns for col in expected_cols
        ), "Missing columns in train features"
        assert (
            "target" in train_features.columns
        ), "Target column missing in train features"
        print(f"Train features shape: {train_features.shape}")
    else:
        print(
            "Warning: Train features empty (possibly no markdown in sampled notebooks)."
        )

    # Check Val
    if not val_features.empty:
        assert all(
            col in val_features.columns for col in expected_cols
        ), "Missing columns in val features"
        print(f"Val features shape: {val_features.shape}")

    # Check Test
    if not test_features.empty:
        assert all(
            col in test_features.columns for col in expected_cols
        ), "Missing columns in test features"
        print(f"Test features shape: {test_features.shape}")

    # -------------------------------------------------------------------------
    # 5. Model Training (LightGBM) Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training Ranker Model...")

    if train_features.empty or val_features.empty:
        print("Skipping training due to empty feature sets.")
        model = None
    else:
        model = train_regressor(train_features, val_features)

        # Verify model file
        assert os.path.exists(Config.LGBM_MODEL_PATH), "LightGBM model file not created"
        print("Ranker model training complete.")

    # -------------------------------------------------------------------------
    # 6. Validation Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 6] Validating Model...")

    if model is not None and not val_features.empty:
        score = validate_model(model, df_val, val_features)
        print(f"Validation Score (Kendall Tau): {score:.4f}")

        assert -1.0 <= score <= 1.0, "Kendall Tau score out of range [-1, 1]"
    else:
        print("Skipping validation.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Submission...")

    if model is not None:
        generate_submission(df_test, test_features, model)

        # Verify submission file
        assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        assert list(sub_df.columns) == [
            "id",
            "cell_order",
        ], "Submission columns incorrect"
        assert len(sub_df) == len(df_test), "Submission row count mismatch"

        print("Submission generated successfully.")
        print(sub_df.head())
    else:
        print("Skipping submission generation.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
