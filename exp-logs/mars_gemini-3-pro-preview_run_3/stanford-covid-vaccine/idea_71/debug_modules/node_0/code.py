import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sys

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, MetricCalculator
from library.data import get_dataloaders, RNADataset
from library.model import HighCapacityAugmentedBiGRU
from library.layers import DecoupledGLUInteraction, PointwiseFFN
from library.train import train_one_epoch, validate, inference


def main():
    print("==== RNA Degradation Prediction Library Demo ====")

    # 1. Configuration Setup
    # Using debug=True to reduce dataset size and epochs for quick demonstration
    print("\n[1] Initializing Configuration...")
    config = Config(debug=True, epochs=1, batch_size=4)

    # Ensure working directory exists (Config does this, but good to verify)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set random seeds for reproducibility
    set_seed(config.SEED)
    print(f"    Device: {config.DEVICE}")
    print(f"    Batch Size: {config.BATCH_SIZE}")
    print(f"    Debug Mode: {config.DEBUG}")

    # 2. Data Loading
    print("\n[2] Loading Data...")
    # This will use the metadata parquet files and create/load .npz caches in ./working
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Verify DataLoader
    print("    Verifying Train Loader...")
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = {"sequence", "bpp_indices", "pair_mask", "targets", "id"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Check dimensions
    # Sequence: (Batch, Seq_Len, Channels=14)
    seq = batch["sequence"]
    bpp = batch["bpp_indices"]
    mask = batch["pair_mask"]
    targets = batch["targets"]
    ids = batch["id"]

    assert seq.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        config.INPUT_CHANNELS,
    ), f"Incorrect sequence shape: {seq.shape}"
    assert bpp.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
    ), f"Incorrect bpp_indices shape: {bpp.shape}"
    assert mask.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
    ), f"Incorrect pair_mask shape: {mask.shape}"
    # Targets: (Batch, Seq_Scored=68, Num_Targets=5)
    assert targets.shape == (
        config.BATCH_SIZE,
        config.SEQ_SCORED,
        config.NUM_TARGETS,
    ), f"Incorrect targets shape: {targets.shape}"

    print("    Batch shapes verified successfully.")

    # 3. Model Instantiation & Layer Verification
    print("\n[3] Instantiating Model...")
    model = HighCapacityAugmentedBiGRU(config)
    model.to(config.DEVICE)

    # Move batch to device
    seq = seq.to(config.DEVICE)
    bpp = bpp.to(config.DEVICE)
    mask = mask.to(config.DEVICE)
    targets = targets.to(config.DEVICE)

    print("    Model initialized.")

    # Test specific layers (Unit Test)
    print("    Verifying DecoupledGLUInteraction Layer...")
    # Hidden dim is 768 in the model (384 * 2)
    hidden_dim = config.HIDDEN_DIM * 2
    interaction_layer = DecoupledGLUInteraction(hidden_dim=hidden_dim).to(config.DEVICE)

    # Create dummy hidden state
    dummy_h = torch.randn(config.BATCH_SIZE, config.SEQ_LENGTH, hidden_dim).to(
        config.DEVICE
    )
    out_h = interaction_layer(dummy_h, bpp, mask)

    assert out_h.shape == dummy_h.shape, "Interaction layer output shape mismatch."
    assert not torch.isnan(out_h).any(), "Interaction layer produced NaNs."
    print("    Layer verification passed.")

    # 4. Forward Pass & Loss Calculation
    print("\n[4] Running Forward Pass & Metric Calculation...")

    # Forward
    preds = model(seq, bpp, mask)

    # Predictions shape should be (Batch, Seq_Len=107, Num_Targets=5)
    # Note: Model predicts for full length 107, but targets are only for first 68.
    assert preds.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        config.NUM_TARGETS,
    ), f"Prediction shape mismatch: {preds.shape}"

    # Metric Calculator
    metric_calc = MetricCalculator(config)

    # Calculate Train Loss (MCRMSE on all 5 targets, sliced to 68)
    loss = metric_calc.compute_train_loss(preds, targets)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert loss.item() > 0, "Loss should be positive."
    print(f"    Forward pass successful. Initial Loss: {loss.item():.4f}")

    # 5. Optimization Step
    print("\n[5] Simulating Optimization Step...")
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Gradients not computed."

    optimizer.step()
    print("    Optimization step completed.")

    # 6. Training Loop Integration
    print("\n[6] Running Training Loop (One Epoch)...")
    # Using the library function train_one_epoch
    epoch_loss = train_one_epoch(
        model, train_loader, optimizer, metric_calc, config.DEVICE, config
    )
    print(f"    Epoch Loss: {epoch_loss:.4f}")

    # 7. Validation Integration
    print("\n[7] Running Validation...")
    # Using the library function validate
    val_score = validate(model, val_loader, metric_calc, config.DEVICE)
    print(f"    Validation MCRMSE: {val_score:.4f}")

    # 8. Inference Integration
    print("\n[8] Running Inference on Test Set...")
    # Using the library function inference
    test_preds, test_ids = inference(model, test_loader, config.DEVICE)

    assert len(test_preds) == len(test_ids), "Mismatch between predictions and IDs."
    # Test preds shape: (N_Samples, 107, 5)
    assert test_preds.shape[1] == config.SEQ_LENGTH, "Inference seq length mismatch."
    assert test_preds.shape[2] == config.NUM_TARGETS, "Inference target dim mismatch."

    print(f"    Inference complete. Generated predictions for {len(test_ids)} samples.")

    # 9. Submission Formatting
    print("\n[9] Formatting Submission...")
    submission_rows = []

    # Process a subset for demonstration
    demo_limit = 5
    print(f"    Formatting first {demo_limit} samples...")

    for i in range(min(len(test_ids), demo_limit)):
        sample_id = test_ids[i]
        sample_preds = test_preds[i]  # (107, 5)

        for seqpos in range(config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()
            submission_rows.append([row_id] + row_values)

    # Convert to DataFrame
    columns = ["id_seqpos"] + config.TARGET_COLS
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    print("    Submission DataFrame Head:")
    print(sub_df.head(3))

    # Check output format
    assert sub_df.shape[1] == 6, "Submission should have 6 columns."
    assert "id_seqpos" in sub_df.columns, "Missing id_seqpos column."

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
