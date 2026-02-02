import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss


def run_demo():
    print("--- Starting RNA Degradation Model Demo ---")

    # 1. Setup Configuration for Demo
    # We create a specific configuration for this run to avoid heavy computation
    config = Config()
    config.working_dir = "./working/demo_run"
    config.epochs = 1
    config.batch_size = 16  # Smaller batch size for demo
    config.hidden_dim = 64  # Smaller model width
    config.n_layers = 2  # Shallower network
    config.embed_dim = 32  # Smaller embeddings

    # Ensure clean working directory
    if os.path.exists(config.working_dir):
        shutil.rmtree(config.working_dir)
    os.makedirs(config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)
    print(f"Configuration set. Working dir: {config.working_dir}")
    print(f"Device: {config.device}")

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # We force processing from source to demonstrate the pipeline,
    # but in a real run, caching (load_cached_data=True) is preferred.
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Verify Data Shapes
    batch = next(iter(train_loader))
    seq, loop, dist, target, mask = batch

    print(
        f"Batch shapes -> Seq: {seq.shape}, Target: {target.shape}, Mask: {mask.shape}"
    )

    # Assertions
    assert seq.shape == (config.batch_size, config.seq_len), "Incorrect sequence shape"
    assert target.shape == (
        config.batch_size,
        config.seq_len,
        3,
    ), "Incorrect target shape"
    assert mask.shape == (config.batch_size, config.seq_len), "Incorrect mask shape"

    # Verify Mask Logic (First 68 positions should be 1.0)
    # Note: The dataset ensures seq_scored is consistent, usually 68.
    assert torch.all(mask[:, :68] == 1.0), "Mask should be 1.0 for first 68 positions"
    # Positions after seq_scored might be 0, but technically the file format allows varying lengths.
    # In this specific dataset, we expect 68 scored.

    print("Data verification passed.")

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    model = RNAModel(config).to(config.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params} parameters.")

    # 4. Forward Pass & Loss
    print("\n--- Forward Pass & Loss Calculation ---")
    criterion = MaskedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # Move batch to device
    seq = seq.to(config.device)
    loop = loop.to(config.device)
    dist = dist.to(config.device)
    target = target.to(config.device)
    mask = mask.to(config.device)

    # Forward
    model.train()
    preds = model(seq, loop, dist)

    # Check output shape
    assert preds.shape == (
        config.batch_size,
        config.seq_len,
        3,
    ), "Output shape mismatch"

    # Calculate Loss
    loss = criterion(preds, target, mask)
    print(f"Initial Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    # 5. Optimization Step (Training Simulation)
    print("\n--- Simulating Training Step ---")

    # Check weights before update (just checking first layer)
    param_before = list(model.parameters())[0].clone()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check weights after update
    param_after = list(model.parameters())[0]
    assert not torch.equal(
        param_before, param_after
    ), "Weights did not update after optimizer step"
    print("Optimizer step successful. Weights updated.")

    # 6. Evaluation Metric
    print("\n--- Calculating Metric (MCRMSE) ---")
    model.eval()
    with torch.no_grad():
        val_preds = model(seq, loop, dist)

    # Calculate MCRMSE only on scored positions (masked)
    # The metric function expects (N, Seq, Channels) or (N, Channels)
    # We should filter by mask or just pass the full tensors if the metric handles it.
    # The provided metric_mcrmse calculates RMSE per column.
    # To be strictly correct with the competition metric, we should mask out unscored positions first
    # or rely on the fact that the target is 0 and prediction might be close to 0,
    # BUT the competition metric is usually computed on the flattened vector of valid positions.

    # For this demonstration using the provided utility:
    # We will manually mask before passing to metric_mcrmse to ensure accuracy
    # Flattening: (Batch * Scored_Len, Channels)

    # Extract valid indices
    valid_mask = mask.bool()

    # Since mask is (B, L), we can't simply index (B, L, 3) directly to get a rectangular matrix
    # if lengths differed, but here they are uniform (68).
    # We can slice:
    scored_len = 68
    y_true_scored = target[:, :scored_len, :]
    y_pred_scored = val_preds[:, :scored_len, :]

    score = metric_mcrmse(y_true_scored, y_pred_scored)
    print(f"MCRMSE Score on batch: {score:.6f}")
    assert score >= 0, "Metric must be non-negative"

    # 7. Inference & Submission Generation
    print("\n--- Generating Submission ---")

    # We'll process one batch from test_loader
    test_batch = next(iter(test_loader))
    t_seq, t_loop, t_dist, _, t_mask = test_batch
    # Note: Test loader returns dummy targets/masks usually, or masks based on seq_scored

    t_seq = t_seq.to(config.device)
    t_loop = t_loop.to(config.device)
    t_dist = t_dist.to(config.device)

    model.eval()
    with torch.no_grad():
        t_preds = model(t_seq, t_loop, t_dist)  # (B, 107, 3)

    t_preds_np = t_preds.cpu().numpy()

    # Prepare submission data
    # We need to map these 3 columns to the 5 required columns:
    # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # The model predicts: reactivity, deg_Mg_pH10, deg_Mg_50C
    # We will fill the missing ones (deg_pH10, deg_50C) with 0 or duplicates.

    # Let's verify IDs
    # The dataloader doesn't return IDs in the batch tuple in the provided `RNADataset.__getitem__`.
    # However, `process_dataframe` returns 'ids'.
    # We can access ids from the dataset directly for this batch if we match indices,
    # but `RNADataset` doesn't return IDs in `__getitem__`.
    # We will simulate the ID retrieval for the demo based on the batch size.

    # In a real inference loop, we would modify __getitem__ to return IDs or iterate sequentially.
    # Here, we just demonstrate formatting the numerical output.

    submission_rows = []
    # Columns in model output: 0:reactivity, 1:deg_Mg_pH10, 2:deg_Mg_50C

    # Columns required in submission:
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i in range(config.batch_size):
        # Dummy ID
        sample_id = f"id_demo_{i}"

        for pos in range(config.seq_len):  # 107
            row_id = f"{sample_id}_{pos}"

            # Get predictions
            reactivity = t_preds_np[i, pos, 0]
            deg_Mg_pH10 = t_preds_np[i, pos, 1]
            deg_Mg_50C = t_preds_np[i, pos, 2]

            # Fill missing with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

    df_sub = pd.DataFrame(submission_rows)
    print(f"Generated submission dataframe with shape: {df_sub.shape}")

    # Verify columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Save to dummy file
    sub_path = os.path.join(config.working_dir, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
