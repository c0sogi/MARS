import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config, set_seed
from library.dataset import get_dataset
from library.model import DeepDecoupledBiGRU
from library.loss_metric import MCRMSELoss, compute_metric
from library.trainer import Trainer


def main():
    print("==== Starting RNA Degradation Pipeline Demo ====")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up configuration...")
    # Initialize config with constraints for speed
    config = Config(
        debug=True,
        epochs=2,
        batch_size=8,
        max_train_samples=100,  # Limit to 100 samples for quick execution
    )

    # Set specific working directory for this demo
    config.working_dir = "./working/demo_execution"
    config.cache_dir = config.working_dir
    config.model_save_path = os.path.join(config.working_dir, "demo_model.pth")
    config.submission_path = os.path.join(config.working_dir, "demo_submission.csv")

    os.makedirs(config.working_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.seed)
    print(f"    Working Directory: {config.working_dir}")
    print(f"    Device: {config.device}")

    # ---------------------------------------------------------
    # 2. Data Loading and Verification
    # ---------------------------------------------------------
    print("\n[2] Loading and verifying datasets...")

    # Load Training Data
    train_dataset = get_dataset("train", config)
    print(f"    Train Dataset Size: {len(train_dataset)}")

    # Assertions to verify data loading logic
    assert (
        len(train_dataset) == config.max_train_samples
    ), f"Expected {config.max_train_samples} samples, got {len(train_dataset)}"

    # Inspect a single sample
    sample = train_dataset[0]
    input_shape = sample["inputs"].shape
    target_shape = sample["targets"].shape

    print(f"    Sample Input Shape: {input_shape} (Expected: (107, 14))")
    print(f"    Sample Target Shape: {target_shape} (Expected: (68, 5))")

    assert input_shape == (107, 14), "Incorrect input feature shape"
    assert target_shape == (68, 5), "Incorrect target shape"
    assert "bpp_indices" in sample and "bpp_mask" in sample, "Missing adjacency info"

    # Load Validation Data (Small subset automatically handled by Config if implemented,
    # but here we rely on the library loading the full val set or cached version)
    val_dataset = get_dataset("val", config)
    print(f"    Val Dataset Size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # ---------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Initializing model and verifying forward pass...")
    model = DeepDecoupledBiGRU(config).to(config.device)

    # Get a batch
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(config.device)
    bpp_indices = batch["bpp_indices"].to(config.device)
    bpp_mask = batch["bpp_mask"].to(config.device)
    targets = batch["targets"].to(config.device)

    # Forward pass
    outputs = model(inputs, bpp_indices, bpp_mask)

    print(f"    Output Shape: {outputs.shape}")

    # Verify output shape: (Batch, Seq_Len, Num_Classes) -> (8, 107, 5)
    expected_shape = (config.batch_size, config.seq_len, config.num_classes)
    assert (
        outputs.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {outputs.shape}"

    # ---------------------------------------------------------
    # 4. Loss and Metric Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss and Metric calculation...")
    loss_fn = MCRMSELoss()

    # Calculate Loss
    loss = loss_fn(outputs, targets)
    print(f"    Batch Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"

    # Calculate Metric
    metric = compute_metric(outputs, targets, config)
    print(f"    Batch MCRMSE Metric: {metric:.4f}")
    assert isinstance(metric, float), "Metric should return a float"

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs)...")
    trainer = Trainer(config, model, train_loader, val_loader)
    trainer.fit()

    # Verify model was saved
    assert os.path.exists(config.model_save_path), "Model file was not saved."
    print("    Training completed and model saved.")

    # ---------------------------------------------------------
    # 6. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission on Test Set...")

    # Load Test Data
    test_dataset = get_dataset("test", config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load best model state
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )
    model.eval()

    preds_list = []
    ids_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(config.device)
            bpp_indices = batch["bpp_indices"].to(config.device)
            bpp_mask = batch["bpp_mask"].to(config.device)
            ids = batch["ids"]

            outputs = model(inputs, bpp_indices, bpp_mask)

            # Move to CPU
            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate predictions: (N_test, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare Submission DataFrame
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    target_cols = (
        config.target_cols
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        # For each sample, we have 107 positions, but usually only up to seq_scored are required?
        # The instructions say: "predict targets for each sequence position... Positions greater than seq_scored are not scored but need a value."
        # We will output all 107 positions as per the output shape.

        sample_preds = all_preds[i]  # Shape (107, 5)

        for seqpos in range(config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    submission_df.to_csv(config.submission_path, index=False)
    print(f"    Submission saved to {config.submission_path}")
    print(f"    Submission Shape: {submission_df.shape}")

    # Final Validation
    assert not submission_df.empty, "Submission DataFrame is empty"
    assert "id_seqpos" in submission_df.columns, "Missing ID column in submission"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
