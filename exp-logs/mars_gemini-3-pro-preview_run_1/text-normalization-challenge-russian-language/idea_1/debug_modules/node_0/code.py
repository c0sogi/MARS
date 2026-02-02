import pandas as pd
import numpy as np
import os
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_processor import get_data
from library.ngram_model import (
    train_model,
    evaluate_model,
    generate_submission,
    HierarchicalLookupModel,
)


def main():
    # 1. Setup and Configuration
    print("--- 1. Setup and Configuration ---")
    seed_everything(42)

    # Configure for a fast demonstration run
    # We use a small subset of data to ensure the script finishes quickly
    Config.MAX_TRAIN_SAMPLES = 5000

    # Redirect working directory to a demo folder to avoid cache conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.TRAIN_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "train_sequences.parquet"
    )
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_sequences.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_sequences.parquet")
    Config.MODEL_STATS_PATH = os.path.join(Config.WORKING_DIR, "ngram_stats.npy")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create the new working directory
    Config.setup()
    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Max train samples set to: {Config.MAX_TRAIN_SAMPLES}")

    # 2. Data Processing
    print("\n--- 2. Data Processing ---")
    # Load and process training data (force re-computation by disabling cache loading)
    print("Processing training data...")
    df_train = get_data("train", load_cached_data=False)

    print("Processing validation data...")
    df_val = get_data("val", load_cached_data=False)

    # Verification
    print(f"Train DataFrame shape: {df_train.shape}")
    print(f"Val DataFrame shape: {df_val.shape}")

    # Check for BOS/EOS tokens
    has_bos = Config.BOS_TOKEN in df_train[Config.INPUT_COL].values
    has_eos = Config.EOS_TOKEN in df_train[Config.INPUT_COL].values

    if not has_bos or not has_eos:
        raise AssertionError("Processed data is missing BOS or EOS tokens.")

    print("Data processing verification passed: BOS/EOS tokens present.")

    # 3. Model Training
    print("\n--- 3. Model Training ---")
    # Train the model
    # load_cached_data=False ensures we actually run the .fit() method
    model = train_model(load_cached_data=False)

    # Verify model learned something
    stats = model.get_stats()
    l1_size = len(stats["l1"])
    print(f"Model trained. Learned {l1_size} unigram mappings.")

    if l1_size == 0:
        raise AssertionError("Model failed to learn any mappings.")

    # 4. Manual Logic Verification (Unit Test)
    print("\n--- 4. Manual Logic Verification ---")
    # We create a small synthetic dataset to verify the hierarchical lookup priority:
    # L3 (Trigram) > L2 (Bigram) > L1 (Unigram)

    # Scenario:
    # Context "A B" -> Target "b_special" (L2 Left)
    # Token "B" -> Target "b_general" (L1)

    # We construct a dataframe simulating processed sequences
    dummy_df = pd.DataFrame(
        {
            Config.SENTENCE_ID_COL: [0, 0, 0, 0, 1, 1, 1, 1],
            Config.TOKEN_ID_COL: [-1, 0, 1, 2, -1, 0, 1, 2],
            Config.INPUT_COL: [
                Config.BOS_TOKEN,
                "A",
                "B",
                Config.EOS_TOKEN,  # Sentence 0
                Config.BOS_TOKEN,
                "C",
                "B",
                Config.EOS_TOKEN,  # Sentence 1
            ],
            Config.TARGET_COL: [
                Config.BOS_TOKEN,
                "a",
                "b_special",
                Config.EOS_TOKEN,
                Config.BOS_TOKEN,
                "c",
                "b_general",
                Config.EOS_TOKEN,
            ],
        }
    )

    # Fit a temporary model
    test_model = HierarchicalLookupModel()
    test_model.fit(dummy_df)

    # Predict
    # We expect the first "B" (preceded by "A") to be "b_special"
    # We expect the second "B" (preceded by "C") to be "b_general" (based on L1 or L2 context)
    preds = test_model.predict(dummy_df)

    # preds list corresponds to non-boundary tokens: A, B, C, B
    expected_preds = ["a", "b_special", "c", "b_general"]

    print(f"Predictions: {preds}")
    print(f"Expected:    {expected_preds}")

    if preds != expected_preds:
        raise AssertionError(
            f"Logic verification failed! Expected {expected_preds}, got {preds}"
        )

    print("Manual logic verification passed.")

    # 5. Evaluation
    print("\n--- 5. Evaluation ---")
    accuracy = evaluate_model(model, df_val)
    print(f"Validation Accuracy: {accuracy:.4f}")

    if not (0.0 <= accuracy <= 1.0):
        raise AssertionError("Accuracy score is out of valid range [0, 1].")

    # 6. Submission Generation
    print("\n--- 6. Submission Generation ---")
    # Note: This processes the full test set (or whatever is in metadata/test.csv)
    # Since we cannot subsample test data via Config easily without modifying library code,
    # we allow this to run. It should be reasonably fast (~1M rows).
    generate_submission(model)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    required_cols = [Config.SUBMISSION_ID_COL, Config.TARGET_COL]
    if not all(col in df_sub.columns for col in required_cols):
        raise AssertionError(f"Submission missing required columns: {required_cols}")

    # Check ID format (e.g., "0_0")
    sample_id = df_sub.iloc[0][Config.SUBMISSION_ID_COL]
    if "_" not in str(sample_id):
        raise AssertionError(f"Submission ID format incorrect. Example: {sample_id}")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
