import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# 1. Suppress TQDM progress bars to meet "silent execution" requirement
# We must patch this before importing the library modules that use tqdm.
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# 2. Import Library Modules
# Note: The library files are assumed to be in ./library based on the prompt.
from library.config import Config
from library.data import load_data, QADataset, get_tokenizer
from library.model import QAModel
from library.utils import jaccard, set_seed
from library import engine


def main():
    print("=== Starting QA Pipeline Demonstration ===")

    # 3. Configure for Speed and Debugging
    # We modify the Config class directly to ensure the demonstration runs quickly.
    print("Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 20  # Use only 20 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Clean working directory to ensure fresh run
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)

    Config.setup()
    set_seed(Config.SEED)

    # 4. Data Pipeline Verification
    print("\n[1/5] Verifying Data Loading and Processing...")
    tokenizer = get_tokenizer()

    # Load training data (debug mode)
    train_df = load_data("train", tokenizer, debug=True)

    # Assertions for DataFrame structure
    assert isinstance(train_df, pd.DataFrame), "load_data should return a DataFrame"
    expected_cols = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing column {col} in processed data"

    print(f"Train DataFrame shape: {train_df.shape}")

    # Test Dataset Class
    train_dataset = QADataset(train_df, mode="train")
    sample_item = train_dataset[0]

    # Assertions for Dataset item
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "start_positions" in sample_item
    assert isinstance(sample_item["input_ids"], torch.Tensor)
    assert sample_item["input_ids"].shape[0] == Config.MAX_LENGTH

    print("Data pipeline verification passed.")

    # 5. Model Verification
    print("\n[2/5] Verifying Model Initialization and Forward Pass...")
    device = Config.DEVICE
    model = QAModel(Config.MODEL_CHECKPOINT)
    model.to(device)
    model.train()

    # Create a dummy batch
    batch_input_ids = sample_item["input_ids"].unsqueeze(0).to(device)
    batch_mask = sample_item["attention_mask"].unsqueeze(0).to(device)
    batch_start = sample_item["start_positions"].unsqueeze(0).to(device)
    batch_end = sample_item["end_positions"].unsqueeze(0).to(device)

    # Forward pass with labels (should return loss)
    loss, start_logits, end_logits = model(
        input_ids=batch_input_ids,
        attention_mask=batch_mask,
        start_positions=batch_start,
        end_positions=batch_end,
    )

    # Assertions for model output
    assert loss is not None, "Model should return loss when labels are provided"
    assert start_logits.shape == (1, Config.MAX_LENGTH), "Incorrect start_logits shape"
    assert end_logits.shape == (1, Config.MAX_LENGTH), "Incorrect end_logits shape"

    # Forward pass without labels (inference mode)
    start_logits_inf, end_logits_inf = model(
        input_ids=batch_input_ids, attention_mask=batch_mask
    )
    assert start_logits_inf.shape == (1, Config.MAX_LENGTH)

    print("Model verification passed.")

    # 6. Training Engine Execution
    print("\n[3/5] Executing Training Engine (Debug Mode)...")
    # This runs the full training loop using the engine.py logic
    engine.run_training(debug=True)

    # Verify artifacts
    model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Training failed to save 'best_model.pth'"
    print(f"Training complete. Model saved to {model_path}")

    # 7. Inference Engine Execution
    print("\n[4/5] Executing Inference Engine (Debug Mode)...")
    # This generates the submission file
    engine.generate_submission(debug=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    assert (
        "id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission missing required columns"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check quoting (heuristic: check if string starts with quote if not empty)
    sample_pred = sub_df["PredictionString"].iloc[0]
    if len(sample_pred) > 0:
        # The requirement says "selected text needs to be quoted".
        # Our engine adds quotes: f'"{pred_text}"'
        assert sample_pred.startswith('"') and sample_pred.endswith(
            '"'
        ), f"PredictionString not properly quoted: {sample_pred}"

    print(f"Submission generated at {Config.SUBMISSION_FILE} with {len(sub_df)} rows.")

    # 8. Metric Verification
    print("\n[5/5] Verifying Metric (Jaccard)...")
    s1 = "This is a test answer"
    s2 = "this is test answer"
    score = jaccard(s1, s2)

    # Intersection: {this, is, test, answer} (len 4)
    # Union: {this, is, a, test, answer} (len 5)
    # Score: 4/5 = 0.8
    assert (
        abs(score - 0.8) < 1e-6
    ), f"Jaccard calculation incorrect. Expected 0.8, got {score}"

    s3 = "Completely different"
    score_zero = jaccard(s1, s3)
    assert score_zero == 0.0, "Jaccard should be 0 for disjoint sets"

    print("Metric verification passed.")

    print("\n=== All Demonstrations and Verifications Completed Successfully ===")


if __name__ == "__main__":
    main()
