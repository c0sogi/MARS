import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import warnings
from transformers import get_linear_schedule_with_warmup

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_optimizer_params
from library.dataset import get_dataloaders
from library.model import ToxicityModel
from library.engine import Engine
from library.metrics import calculate_final_score, calculate_roc_auc


def main():
    # --------------------------------------------------------------------------
    # 0. Setup and Configuration Overrides
    # --------------------------------------------------------------------------
    print("=== Setting up Configuration ===")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override Config for a fast demonstration
    Config.DEBUG = True  # Limits data to 1000 rows
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8  # Smaller batch for safety/speed in demo
    Config.VALID_BATCH_SIZE = 16
    Config.WORKING_DIR = "./working/demo_run"  # Separate dir for demo cache

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 1. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n=== Verifying Data Loading ===")

    # Generate DataLoaders (this triggers process_data which tokenizes and caches)
    # load_cached_data=False forces reprocessing to demonstrate the pipeline
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify DataLoaders
    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches:   {len(val_loader)}")
    print(f"Test Batches:  {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."

    # Inspect a single batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targets = batch["targets"]

    print(f"Batch Input Shape: {input_ids.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions for shapes
    # Input: [Batch, Max_Len]
    assert input_ids.shape[0] == Config.TRAIN_BATCH_SIZE
    assert input_ids.shape[1] == Config.MAX_LEN
    # Targets: [Batch, 1 + Num_Aux] (1 main target + 9 identity columns)
    expected_target_cols = 1 + len(Config.IDENTITY_COLUMNS)
    assert targets.shape[1] == expected_target_cols

    print("Data Loading Verified.")

    # --------------------------------------------------------------------------
    # 2. Model Initialization Verification
    # --------------------------------------------------------------------------
    print("\n=== Verifying Model Initialization ===")

    model = ToxicityModel()
    model.to(Config.DEVICE)

    # Verify model structure (simple check of head dimensions)
    assert model.linear.out_features == Config.NUM_LABELS
    assert model.aux_linear.out_features == Config.NUM_AUX_LABELS

    # Verify Forward Pass with dummy data
    # Move batch to device
    b_input_ids = input_ids.to(Config.DEVICE)
    b_mask = attention_mask.to(Config.DEVICE)

    with torch.no_grad():
        tox_logits, aux_logits = model(b_input_ids, b_mask)

    print(f"Output Logits Shape: {tox_logits.shape}")
    assert tox_logits.shape == (Config.TRAIN_BATCH_SIZE, 1)
    assert aux_logits.shape == (Config.TRAIN_BATCH_SIZE, Config.NUM_AUX_LABELS)

    print("Model Initialization Verified.")

    # --------------------------------------------------------------------------
    # 3. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n=== Verifying Training Loop ===")

    # Setup Optimizer with LLRD
    optimizer_params = get_optimizer_params(
        model,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )
    optimizer = optim.AdamW(optimizer_params)

    # Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Initialize Engine
    engine = Engine(model, optimizer, scheduler, Config.DEVICE)

    # Train for 1 epoch
    print("Training for 1 epoch (Debug subset)...")
    train_loss = engine.train_one_epoch(train_loader, epoch_index=1)

    assert not np.isnan(train_loss), "Training loss is NaN."
    print(f"Training Loss: {train_loss:.4f}")

    print("Training Loop Verified.")

    # --------------------------------------------------------------------------
    # 4. Validation and Metric Verification
    # --------------------------------------------------------------------------
    print("\n=== Verifying Validation & Metrics ===")

    # Run validation
    val_loss, val_score = engine.validate(val_loader)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Score: {val_score:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN."
    # Score might be 0.5 or similar if model hasn't learned much in 1 debug epoch,
    # but it should be a valid float.
    assert 0.0 <= val_score <= 1.0, "Validation score out of range."

    # Verify Metric Calculation Logic explicitly with dummy data
    # Create a dummy dataframe mimicking validation output
    N = 100
    dummy_df = pd.DataFrame(
        {
            "target": np.random.choice([0.0, 1.0], N),
            "prediction": np.random.rand(N),
            # Add a few identity columns
            "male": np.random.choice([0.0, 1.0], N),
            "female": np.random.choice([0.0, 1.0], N),
        }
    )

    # Temporarily mock Config.IDENTITY_COLUMNS for this specific check
    # to ensure calculate_final_score doesn't fail if columns are missing in dummy df
    original_identities = Config.IDENTITY_COLUMNS
    Config.IDENTITY_COLUMNS = ["male", "female"]

    score_dummy, details = calculate_final_score(dummy_df)

    # Restore Config
    Config.IDENTITY_COLUMNS = original_identities

    print(f"Dummy Metric Check Score: {score_dummy:.4f}")
    assert "overall_auc" in details
    assert "subgroup_auc_mean" in details

    print("Validation & Metrics Verified.")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission Verification
    # --------------------------------------------------------------------------
    print("\n=== Verifying Inference & Submission ===")

    # We manually run inference using the test loader
    model.eval()
    test_preds = []
    test_ids = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            ids = batch["ids"]

            # Trim
            input_ids, attention_mask = engine._trim_tensors(input_ids, attention_mask)

            # Forward
            tox_logits, _ = model(input_ids, attention_mask)
            preds = torch.sigmoid(tox_logits).cpu().numpy().flatten()

            test_preds.extend(preds)
            test_ids.extend(ids.numpy())

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "prediction": test_preds})

    print(f"Submission Shape: {submission.shape}")
    print(submission.head())

    # Verify against sample submission logic
    # In Debug mode, we only have a subset of test data (1000 rows)
    # The real submission file would need all rows, but for this demo,
    # we verify that we generated predictions for the loaded data.
    assert len(submission) == len(test_loader.dataset)
    assert submission["prediction"].min() >= 0.0
    assert submission["prediction"].max() <= 1.0

    # Save submission (mock)
    submission_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved demo submission to {submission_path}")

    print("Inference Verified.")
    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
