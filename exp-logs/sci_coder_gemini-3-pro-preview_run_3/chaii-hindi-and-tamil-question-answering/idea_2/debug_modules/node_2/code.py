import os
import shutil
import torch
import pandas as pd
import warnings
from transformers import logging as transformers_logging

# Import library components
from library.config import Config
from library.utils import set_seed, jaccard, compute_score
from library.data_loader import get_dataloaders
from library.model import get_model
from library.trainer import train_runner
from library.inference import run_inference_pipeline


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("=== Setting up Demo Configuration ===")

    # Suppress warnings and logs for cleaner output
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override Config for a fast demonstration
    # We use a separate directory to avoid interfering with other runs
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce training load for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.SEEDS = [42]  # Run only one seed

    # Ensure clean directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Metric Verification
    # -------------------------------------------------------------------------
    print("\n=== Verifying Metric Implementation ===")
    s1 = "hello world"
    s2 = "hello world"
    s3 = "hello python"

    score_perfect = jaccard(s1, s2)
    score_partial = jaccard(s1, s3)

    print(f"Jaccard('{s1}', '{s2}') = {score_perfect}")
    print(f"Jaccard('{s1}', '{s3}') = {score_partial}")

    assert score_perfect == 1.0, "Jaccard score for identical strings should be 1.0"
    assert (
        0.0 < score_partial < 1.0
    ), "Jaccard score for partial match should be between 0 and 1"

    avg_score = compute_score([s1, s1], [s2, s3])
    print(f"Average Score: {avg_score}")
    assert avg_score == (1.0 + score_partial) / 2, "Compute score calculation incorrect"

    # -------------------------------------------------------------------------
    # 3. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n=== Verifying Data Loading (Debug Mode) ===")
    # get_dataloaders(debug=True) processes a small subset (20 rows) and caches it.
    # We disable loading existing cache initially to force processing of the debug subset.
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")
    print(f"Test Batches: {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty"

    # Inspect a single batch to verify structure
    batch = next(iter(train_loader))
    required_keys = ["input_ids", "attention_mask", "labels", "example_id"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    print("Batch keys verified.")
    print(f"Input IDs Shape: {batch['input_ids'].shape}")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n=== Verifying Model Architecture ===")
    model = get_model()
    model.train()

    # Move batch to device for testing
    device = Config.DEVICE
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Perform a forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert hasattr(outputs, "loss"), "Model output missing loss"
    assert hasattr(outputs, "logits"), "Model output missing logits"
    print(f"Forward pass successful. Loss: {outputs.loss.item():.4f}")

    # Cleanup model to free memory for the next steps
    del model, outputs, input_ids, attention_mask, labels
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Training Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n=== Executing Training Pipeline (Seed 42) ===")
    # train_runner will use the cached debug data generated in step 3.
    # It saves the model to Config.WORKING_DIR/model_seed_42.pt
    train_runner(seed=42, debug=True)

    model_path = os.path.join(Config.WORKING_DIR, "model_seed_42.pt")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print(f"Model saved successfully at {model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n=== Executing Inference Pipeline ===")
    # run_inference_pipeline loads the trained model and the test data cache.
    # It generates the submission file.
    run_inference_pipeline()

    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(submission_path), "Submission file not generated"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")
    print("Head of submission:")
    print(df_sub.head())

    # Verify submission format
    required_cols = ["id", "PredictionString"]
    for col in required_cols:
        assert col in df_sub.columns, f"Submission missing column: {col}"

    # Verify we have predictions for the debug test set
    assert len(df_sub) > 0, "Submission file is empty"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
