import sys
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library modules
# We assume the file structure is preserved as described in the prompt
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import SiameseDeberta
from library.engine import train_fn, eval_fn


def run_demo():
    print("==== Starting Self-Contained Demo ====")

    # 1. Configuration Overrides
    # We modify the Config class attributes directly to optimize for a quick demonstration.
    print("[1] Configuring environment for fast execution...")
    Config.DEBUG = True  # Subsamples data (100 train, 50 val, 50 test)
    Config.MAX_LENGTH = 64  # Reduces tokenization length for speed
    Config.TRAIN_BATCH_SIZE = 4  # Small batch size
    Config.VALID_BATCH_SIZE = 8
    Config.EPOCHS = 1  # Single epoch
    Config.ACCUMULATION_STEPS = 1  # Update every step
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 2. Data Preparation
    print("[2] Loading Data and Tokenizer...")
    # Initialize tokenizer (DeBERTa-v3-large tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Get DataLoaders
    # We disable caching to ensure we read fresh from the metadata CSVs for this demo
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    # Verification: Check Data Structure
    print("    Verifying DataLoader batch structure...")
    try:
        batch = next(iter(train_loader))

        # Check for required keys
        required_keys = [
            "input_ids_a",
            "attention_mask_a",
            "response_mask_a",
            "input_ids_b",
            "attention_mask_b",
            "response_mask_b",
            "scalars",
            "target",
        ]
        for key in required_keys:
            assert key in batch, f"Missing key in batch: {key}"

        # Check dimensions
        # input_ids: (Batch, SeqLen)
        assert batch["input_ids_a"].shape == (
            Config.TRAIN_BATCH_SIZE,
            Config.MAX_LENGTH,
        )
        # scalars: (Batch, 3) -> [log_prompt, log_resp_a, log_resp_b]
        assert batch["scalars"].shape == (Config.TRAIN_BATCH_SIZE, 3)
        # target: (Batch, 3) -> [win_a, win_b, tie]
        assert batch["target"].shape == (Config.TRAIN_BATCH_SIZE, 3)

        print("    DataLoader verification passed.")
    except StopIteration:
        raise ValueError("Train DataLoader is empty!")

    # 3. Model Initialization
    print("[3] Initializing SiameseDeberta Model...")
    device = Config.DEVICE
    model = SiameseDeberta()
    model.to(device)

    # Verification: Forward Pass
    print("    Verifying Model Forward Pass...")
    model.eval()
    with torch.no_grad():
        # Prepare inputs on device
        inputs = {k: v.to(device) for k, v in batch.items() if k != "target"}
        logits = model(**inputs)

    # Check output shape: (Batch, NumClasses)
    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.TRAIN_BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    # Check for NaNs
    assert not torch.isnan(logits).any(), "Model produced NaN logits"
    print("    Model forward pass verified.")

    # 4. Training Loop Demonstration
    print("[4] Executing Training Loop (1 Epoch)...")

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Train
    avg_loss = train_fn(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=1,
    )
    print(f"    Training complete. Average Loss: {avg_loss:.6f}")

    # Basic sanity check on loss
    assert avg_loss > 0, "Training loss should be positive."
    assert avg_loss < 10, "Training loss is unusually high, check initialization."

    # 5. Evaluation Demonstration
    print("[5] Evaluating on Validation Set...")
    val_results = eval_fn(model, val_loader, device)

    # Verify results structure
    assert "loss" in val_results
    assert "metrics" in val_results
    assert "log_loss" in val_results["metrics"]

    print(f"    Validation Log Loss: {val_results['metrics']['log_loss']:.6f}")

    # 6. Inference and Submission
    print("[6] Generating Predictions on Test Set...")
    test_results = eval_fn(model, test_loader, device)
    preds = test_results["predictions"]

    # Verify predictions
    # DEBUG mode limits test set to 50 rows
    expected_rows = 50
    assert preds.shape == (
        expected_rows,
        3,
    ), f"Prediction shape mismatch. Expected ({expected_rows}, 3), got {preds.shape}"

    # Verify probability constraints (Sum to 1)
    row_sums = preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Predictions do not sum to 1.0"

    print("    Predictions generated successfully.")

    # Create Submission File
    print("[7] Creating Submission File...")
    # Load the test IDs corresponding to the debug subset
    # Note: get_dataloaders uses head(50) when DEBUG=True
    test_df_subset = pd.read_csv(Config.TEST_PATH).head(expected_rows)

    submission = pd.DataFrame(
        {
            "id": test_df_subset["id"],
            "winner_model_a": preds[:, 0],
            "winner_model_b": preds[:, 1],
            "winner_tie": preds[:, 2],
        }
    )

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)

    # Verify file existence and content
    assert os.path.exists(submission_path)
    saved_df = pd.read_csv(submission_path)
    assert len(saved_df) == expected_rows
    assert list(saved_df.columns) == [
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ]

    print(f"    Submission saved to: {submission_path}")
    print("==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
