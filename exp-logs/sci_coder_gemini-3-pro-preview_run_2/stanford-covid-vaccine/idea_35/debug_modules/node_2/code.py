import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import SF_DCN
from library.train import (
    train_one_epoch,
    validate,
    generate_submission,
    masked_mcrmse_loss,
)


def main():
    print("=== Starting Demonstration of RNA Degradation Prediction Library ===")

    # 1. Configuration
    # We enable debug mode to reduce batch size and set epochs to 1 for a quick run.
    # We also specify a custom working directory for this demo.
    print("\n[1] Initializing Configuration...")
    config = Config(debug=True, epochs=1, working_dir="./working/demo_execution")
    os.makedirs(config.working_dir, exist_ok=True)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Batch Size: {config.batch_size}")
    print(f"    Epochs: {config.epochs}")

    # 2. Data Loading
    # We set load_cached_data=False to demonstrate processing from raw metadata CSVs.
    print("\n[2] Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Verification of Data Shapes
    print("    Verifying Data Shapes from Train Loader...")
    inputs, partner_idx, targets, _ = next(iter(train_loader))

    # Inputs: (Batch, Channels, SeqLen)
    # Channels = 18 (4 Seq + 3 Struct + 7 Loop + 4 PartnerID)
    # SeqLen = 107
    assert inputs.ndim == 3, f"Expected 3D inputs, got {inputs.ndim}"
    assert (
        inputs.shape[1] == config.input_channels
    ), f"Expected {config.input_channels} channels, got {inputs.shape[1]}"
    assert (
        inputs.shape[2] == config.seq_length
    ), f"Expected sequence length {config.seq_length}, got {inputs.shape[2]}"

    # Partner Indices: (Batch, SeqLen)
    assert partner_idx.ndim == 2, f"Expected 2D partner_idx, got {partner_idx.ndim}"
    assert (
        partner_idx.shape[1] == config.seq_length
    ), f"Expected length {config.seq_length}, got {partner_idx.shape[1]}"

    # Targets: (Batch, SeqLen, 5)
    assert targets.ndim == 3, f"Expected 3D targets, got {targets.ndim}"
    assert (
        targets.shape[1] == config.seq_length
    ), f"Expected length {config.seq_length}, got {targets.shape[1]}"
    assert targets.shape[2] == 5, f"Expected 5 target columns, got {targets.shape[2]}"

    print("    Data shapes verified successfully.")

    # 3. Model Initialization
    print("\n[3] Initializing SF_DCN Model...")
    model = SF_DCN(config).to(device)

    # Verification of Model Forward Pass
    print("    Verifying Forward Pass...")
    inputs = inputs.to(device)
    partner_idx = partner_idx.to(device)

    # The model returns predictions from both pass 1 and pass 2
    with torch.no_grad():
        y1, y2 = model(inputs, partner_idx)

    # Check outputs
    assert y1.shape == (
        inputs.shape[0],
        config.seq_length,
        5,
    ), f"Output y1 shape mismatch: {y1.shape}"
    assert y2.shape == (
        inputs.shape[0],
        config.seq_length,
        5,
    ), f"Output y2 shape mismatch: {y2.shape}"
    print("    Forward pass verified. Output shapes match expected dimensions.")

    # 4. Loss Calculation
    print("\n[4] Verifying Loss Calculation...")
    targets = targets.to(device)

    # Calculate loss on the scored portion of the sequence
    loss = masked_mcrmse_loss(y2, targets, config.scored_indices, config.seq_scored)

    assert isinstance(loss.item(), float), "Loss should be a float scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"    Loss calculated successfully: {loss.item():.6f}")

    # 5. Training Loop Simulation
    print("\n[5] Running Training Simulation (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
    print(f"    Epoch 1 Train Loss: {train_loss:.6f}")
    assert train_loss > 0, "Train loss should be positive"

    # 6. Validation
    print("\n[6] Running Validation...")
    val_score = validate(model, val_loader, config, device)
    print(f"    Validation MCRMSE: {val_score:.6f}")
    assert val_score >= 0, "Validation score should be non-negative"

    # 7. Submission Generation
    print("\n[7] Generating Submission...")
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    generate_submission(model, test_loader, config, device)

    # Verify the generated file
    assert os.path.exists(config.submission_path), "Submission file was not created"

    df_sub = pd.read_csv(config.submission_path)
    print(f"    Loaded submission file with shape: {df_sub.shape}")

    # Verify Columns
    expected_cols = ["id_seqpos"] + config.target_cols
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Column mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Verify Rows
    # Test set has 240 samples. Each sample has 107 positions.
    # Total rows should be 240 * 107 = 25680
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print("    Submission format verified successfully.")
    print("\n=== All demonstrations completed successfully ===")


if __name__ == "__main__":
    main()
