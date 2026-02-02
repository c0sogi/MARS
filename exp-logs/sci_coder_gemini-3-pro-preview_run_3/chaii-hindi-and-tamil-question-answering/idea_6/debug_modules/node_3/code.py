import os
import pandas as pd
import torch
import shutil
from library.configuration import Config
from library.utilities import jaccard, compute_average_jaccard, set_seed
from library.tapt_engine import run_tapt_pretraining
from library.qa_data_processing import get_qa_data
from library.qa_training_engine import run_qa_training
from library.qa_inference_engine import generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config parameters to run on a tiny subset of data
    Config.MAX_TRAIN_SAMPLES = 50  # Only use 50 samples for training
    Config.MAX_VAL_SAMPLES = 20  # Only use 20 samples for validation

    # Reduce training duration
    Config.TAPT_EPOCHS = 1
    Config.EPOCHS = 1

    # Reduce batch sizes for the demo
    Config.TAPT_BATCH_SIZE = 4
    Config.TRAIN_BATCH_SIZE = 4
    Config.EVAL_BATCH_SIZE = 8

    # Run only one seed to save time (disable ensembling for demo)
    Config.SEEDS = [42]

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Demonstrate Utilities
    # --------------------------------------------------------------------------
    print("\n[2] Demonstrating Utility Functions...")

    # Test Jaccard
    s1 = "apple banana orange"
    s2 = "apple banana"
    score = jaccard(s1, s2)
    # Intersection: {apple, banana} (2), Union: {apple, banana, orange} (3) -> 2/3
    expected_score = 2.0 / 3.0
    print(f"Jaccard('{s1}', '{s2}') = {score:.4f}")
    assert abs(score - expected_score) < 1e-6, "Jaccard calculation incorrect"

    # Test Average Jaccard
    gts = ["a b", "c d"]
    preds = ["a b", "c e"]
    # Scores: 1.0 and 0.333... -> Avg: 0.666...
    avg_score = compute_average_jaccard(gts, preds)
    print(f"Average Jaccard: {avg_score:.4f}")
    assert avg_score > 0.0, "Average Jaccard should be positive"

    # --------------------------------------------------------------------------
    # 3. Demonstrate Task-Adaptive Pretraining (TAPT)
    # --------------------------------------------------------------------------
    print("\n[3] Running Task-Adaptive Pretraining (TAPT)...")

    # This will train a Masked Language Model on the context text
    # We force load_cached_data=False to demonstrate the pipeline running from scratch
    run_tapt_pretraining(load_cached_data=False)

    # Verify output
    # Check for config.json as a proxy for successful model saving
    assert os.path.exists(
        os.path.join(Config.TAPT_OUTPUT_DIR, "config.json")
    ), "TAPT model config not found."
    print("TAPT completed successfully.")

    # --------------------------------------------------------------------------
    # 4. Demonstrate QA Data Processing
    # --------------------------------------------------------------------------
    print("\n[4] Processing QA Data...")

    # This generates features (input_ids, attention_mask, labels)
    # It will use the tokenizer from the TAPT step above
    train_ds, val_ds, test_ds = get_qa_data(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")
    print(f"Test Dataset Size: {len(test_ds)}")

    # Verify dataset content
    if len(train_ds) > 0:
        sample_item = train_ds[0]
        assert "input_ids" in sample_item, "Dataset item missing input_ids"
        assert "attention_mask" in sample_item, "Dataset item missing attention_mask"
        assert "labels" in sample_item, "Dataset item missing labels"
        assert isinstance(
            sample_item["input_ids"], torch.Tensor
        ), "input_ids should be a Tensor"

    print("QA Data processing verified.")

    # --------------------------------------------------------------------------
    # 5. Demonstrate QA Training
    # --------------------------------------------------------------------------
    print("\n[5] Running QA Training...")

    # This trains the token classification model
    # It uses the TAPT model as a base if found
    run_qa_training(load_cached_data=True)

    # Verify model checkpoint
    expected_model_path = os.path.join(Config.QA_OUTPUT_DIR, "model_seed_42.pt")
    assert os.path.exists(
        expected_model_path
    ), f"Model checkpoint not found at {expected_model_path}"
    print(f"QA Model trained and saved to {expected_model_path}")

    # --------------------------------------------------------------------------
    # 6. Demonstrate Inference & Submission Generation
    # --------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # This runs inference on the test set and generates submission.csv
    generate_submission()

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not created"

    # Validate content format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print("First 3 rows:")
    print(df_sub.head(3))

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        "PredictionString" in df_sub.columns
    ), "Submission missing 'PredictionString' column"

    # Check that we have predictions for the test set
    # Note: Since we limited data via Config.MAX_TRAIN_SAMPLES, the test set processing
    # in get_qa_data also respects this limit for consistency in debugging.
    # The actual test set is 112 rows. We expect min(50, 112) = 50 rows.
    expected_rows = min(Config.MAX_TRAIN_SAMPLES, 112)
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
