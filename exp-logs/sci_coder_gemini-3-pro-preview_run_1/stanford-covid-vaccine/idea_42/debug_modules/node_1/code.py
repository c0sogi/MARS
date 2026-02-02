import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_loader
from library.model import RNAModel
from library.engine import train_one_epoch, validate


def run_demo():
    # 1. Setup and Configuration Overrides for Demo
    print("--- 1. Setup and Configuration ---")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set deterministic seed
    set_seed(42)

    # Override Config for a quick demonstration run
    Config.epochs = 2
    Config.batch_size = 8
    Config.working_dir = "./working/demo_run"
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Ensure demo directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"Device: {Config.device}")
    print(f"Demo Working Directory: {Config.working_dir}")

    # 2. Metric Verification
    print("\n--- 2. Verifying Metric Logic (MCRMSE) ---")
    # Create random dummy data: (Batch=10, Seq_Len=68, Targets=3)
    # Note: The metric handles slicing internally if seq_len > 68, but we test exact match here.
    y_true = torch.rand(10, 68, 3)
    y_pred = torch.rand(10, 68, 3)

    # Calculate using library function
    lib_loss = mcrmse_loss(y_true, y_pred)

    # Calculate manually
    mse_per_col = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))
    rmse_per_col = torch.sqrt(mse_per_col)
    manual_loss = torch.mean(rmse_per_col)

    print(f"Library Loss: {lib_loss.item():.6f}")
    print(f"Manual Loss:  {manual_loss.item():.6f}")

    # Assert correctness
    assert torch.isclose(
        lib_loss, manual_loss, atol=1e-6
    ), "Metric calculation mismatch!"
    print("Metric verification passed.")

    # 3. Data Loading Demonstration
    print("\n--- 3. Data Loading (Subset) ---")
    # Load a small subset of training data (max_samples=32)
    train_loader = get_loader(
        "train",
        batch_size=Config.batch_size,
        shuffle=False,
        max_samples=32,
        load_cached_data=False,  # Force reload from metadata for demo purposes
    )

    # Fetch one batch to inspect
    batch = next(iter(train_loader))

    # Verify keys and shapes
    seq = batch["sequence"]
    targets = batch["targets"]
    pair_offset = batch["pair_offset"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Sequence shape: {seq.shape} (Expected: [{Config.batch_size}, 107])")
    print(f"Targets shape:  {targets.shape} (Expected: [{Config.batch_size}, 107, 3])")

    assert seq.shape == (Config.batch_size, 107), "Incorrect sequence shape"
    assert targets.shape == (Config.batch_size, 107, 3), "Incorrect targets shape"
    assert pair_offset.shape == (Config.batch_size, 107), "Incorrect pair_offset shape"
    print("Data loading verification passed.")

    # 4. Model Instantiation and Forward Pass
    print("\n--- 4. Model Initialization & Forward Pass ---")
    model = RNAModel().to(Config.device)

    # Move batch to device
    seq = seq.to(Config.device)
    loop = batch["loop_type"].to(Config.device)
    pair = batch["pair_offset"].to(Config.device)

    # Forward pass
    outputs = model(seq, loop, pair)

    print(f"Output shape: {outputs.shape} (Expected: [{Config.batch_size}, 107, 3])")

    # Assert output shape matches expectations
    assert outputs.shape == (Config.batch_size, 107, 3), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN values"
    print("Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    print("\n--- 5. Training Loop Execution (2 Epochs) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    criterion = nn.MSELoss()

    for epoch in range(Config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.device)
        print(f"Epoch {epoch+1} Loss: {loss:.6f}")

        # Assert loss is valid
        assert loss > 0, "Training loss should be positive"
        assert not np.isnan(loss), "Training loss is NaN"

    # Save the model for inference step
    torch.save(model.state_dict(), Config.model_save_path)
    print("Model saved successfully.")

    # 6. Validation Demonstration
    print("\n--- 6. Validation Execution ---")
    val_loader = get_loader(
        "val",
        batch_size=Config.batch_size,
        max_samples=32,
        shuffle=False,
        load_cached_data=False,
    )
    val_score = validate(model, val_loader, Config.device)

    print(f"Validation MCRMSE: {val_score:.6f}")
    assert val_score > 0, "Validation score should be positive"

    # 7. Inference and Submission Generation
    print("\n--- 7. Inference & Submission Generation ---")
    # Load test data (subset)
    test_loader = get_loader(
        "test",
        batch_size=Config.batch_size,
        max_samples=16,
        shuffle=False,
        load_cached_data=False,
    )

    # Load best model
    model.load_state_dict(
        torch.load(Config.model_save_path, map_location=Config.device)
    )
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(Config.device)
            loop = batch["loop_type"].to(Config.device)
            pair = batch["pair_offset"].to(Config.device)
            ids = batch["id"]

            preds = model(seq, loop, pair)

            ids_list.extend(ids)
            preds_list.append(preds.cpu().numpy())

    preds_array = np.concatenate(preds_list, axis=0)
    print(f"Inference complete. Predictions shape: {preds_array.shape}")

    # Format Submission
    submission_data = []
    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_array[i]
        for j in range(Config.seq_len):
            row_id = f"{sample_id}_{j}"

            # Extract scored columns
            reactivity = float(sample_preds[j, 0])
            deg_Mg_pH10 = float(sample_preds[j, 1])
            deg_Mg_50C = float(sample_preds[j, 2])

            # Fill unscored columns
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    df_sub = pd.DataFrame(
        submission_data,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )

    # Verify submission structure
    expected_rows = len(ids_list) * Config.seq_len
    print(f"Submission rows: {len(df_sub)} (Expected: {expected_rows})")
    assert len(df_sub) == expected_rows, "Submission row count mismatch"

    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
