import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SharedBottomRoBERTa
from library.engine import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    # Initialize Config with debug=True to use a small subset (100 samples)
    # Set epochs=1 and batch_size=4 for rapid execution
    print("\n[Step 1] Initializing Configuration...")
    config = Config(debug=True, batch_size=4, epochs=1)
    config.print_config()

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading...")
    # We manually get dataloaders here to inspect a batch before training
    loaders = get_dataloaders(config, load_cached_data=True)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Check if loader has data
    assert len(train_loader) > 0, "Train loader should not be empty."

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = [
        "qa_id",
        "input_ids_q",
        "attention_mask_q",
        "input_ids_a",
        "attention_mask_a",
        "labels",
    ]
    for key in expected_keys:
        assert key in batch, f"Batch is missing key: {key}"

    # Verify shapes
    # Batch size is 4, Sequence length is up to 512, Targets is 30
    batch_size = batch["input_ids_q"].size(0)
    assert (
        batch_size == config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {config.BATCH_SIZE}, got {batch_size}"
    assert batch["labels"].shape == (
        batch_size,
        30,
    ), f"Labels shape mismatch. Expected ({batch_size}, 30), got {batch['labels'].shape}"

    print(
        f"    Batch verification passed. Input shape: {batch['input_ids_q'].shape}, Labels shape: {batch['labels'].shape}"
    )

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")
    model = SharedBottomRoBERTa(config)
    model.to(config.device)

    # Perform a forward pass with the fetched batch
    print("    Running forward pass check...")
    with torch.no_grad():
        input_ids_q = batch["input_ids_q"].to(config.device)
        attention_mask_q = batch["attention_mask_q"].to(config.device)
        input_ids_a = batch["input_ids_a"].to(config.device)
        attention_mask_a = batch["attention_mask_a"].to(config.device)

        logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)

    # Verify output shape (Batch_Size, Num_Targets)
    assert logits.shape == (
        batch_size,
        30,
    ), f"Model output shape mismatch. Expected ({batch_size}, 30), got {logits.shape}"
    print("    Model forward pass successful. Output shape verified.")

    # ---------------------------------------------------------
    # 4. Training Execution
    # ---------------------------------------------------------
    print("\n[Step 4] Running Training Loop...")
    # run_training handles the full loop: train, validate, and save best_model.pth
    run_training(config)

    # Verify model was saved
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), f"Model file not found at {config.MODEL_SAVE_PATH}"
    print("    Training completed and model saved successfully.")

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n[Step 5] Running Inference on Test Set...")

    # Load the best saved model
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.device)
    )
    model.eval()

    all_preds = []
    all_qa_ids = []

    print("    Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids_q = batch["input_ids_q"].to(config.device)
            attention_mask_q = batch["attention_mask_q"].to(config.device)
            input_ids_a = batch["input_ids_a"].to(config.device)
            attention_mask_a = batch["attention_mask_a"].to(config.device)

            # Forward pass
            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)

            # Apply Sigmoid to get probabilities [0, 1]
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_qa_ids.extend(batch["qa_id"].numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Create Submission DataFrame
    # Get target column names from sample submission
    sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH)
    target_cols = [c for c in sample_sub.columns if c != "qa_id"]

    submission_df = pd.DataFrame(all_preds, columns=target_cols)
    submission_df.insert(0, "qa_id", all_qa_ids)

    # Save to disk
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {config.SUBMISSION_PATH}")

    # ---------------------------------------------------------
    # 6. Final Validation
    # ---------------------------------------------------------
    print("\n[Step 6] Validating Submission File...")

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file does not exist."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    # Check dimensions
    # Test set size in DEBUG mode is 100 (SUBSET_SIZE) or less if file is smaller
    # The actual test.csv has 608 rows.
    # If DEBUG=True, get_dataloaders subsets the dataframe to 100.
    expected_rows = min(config.SUBSET_SIZE, 608)
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert (
        df_sub.shape[1] == 31
    ), f"Submission column count mismatch. Expected 31, got {df_sub.shape[1]}"

    # Check value range
    pred_values = df_sub.iloc[:, 1:].values
    assert pred_values.min() >= 0.0, "Found probabilities < 0.0"
    assert pred_values.max() <= 1.0, "Found probabilities > 1.0"

    print("    Submission file validation passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
