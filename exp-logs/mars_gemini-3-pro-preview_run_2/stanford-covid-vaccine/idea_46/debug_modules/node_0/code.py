import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure the library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.data import process_data, RNADataset
from library.model import SDFRNModel
from library.loss import mcrmse_loss
from library.train import train_one_epoch, validate


def main():
    print("Initializing Demo Script...")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly to create a lightweight model and training loop.
    print("Configuring lightweight parameters...")
    Config.CACHE_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    # Training Hyperparameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Model Architecture (Reduced for speed)
    Config.BACKBONE_LAYERS = [1, 2]  # Reduced from [1, 2, 4, 8, 16, 32]
    Config.FEEDBACK_LAYERS = [1]  # Reduced from [1, 2, 4]
    Config.EMBED_DIM = 16  # Reduced from 32
    Config.LATENT_DIM = 32  # Reduced from 64
    Config.RNN_HIDDEN = 32  # Reduced from 64

    # Ensure clean state for the demo cache
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Processing and Loading
    print("\n--- Data Processing ---")
    # We load the metadata using the library function.
    # load_cached_data=False forces reprocessing to demonstrate the pipeline.
    full_train_data = process_data("train", load_cached_data=False)
    full_val_data = process_data("val", load_cached_data=False)

    # Helper to create a mini subset for speed
    def create_subset(data_dict, size=16):
        subset = {}
        for k, v in data_dict.items():
            subset[k] = v[:size]
        return subset

    # Create mini datasets
    mini_train_data = create_subset(full_train_data, size=16)
    mini_val_data = create_subset(full_val_data, size=8)

    print(f"Mini Train Data IDs: {len(mini_train_data['ids'])}")
    print(f"Mini Val Data IDs: {len(mini_val_data['ids'])}")

    # Instantiate Datasets
    train_dataset = RNADataset(mini_train_data, mode="train")
    val_dataset = RNADataset(mini_val_data, mode="val")

    # Verify Dataset Item Structure
    sample_item = train_dataset[0]
    required_keys = ["seq", "struct", "loop", "partner_idx", "partner_id", "targets"]
    for key in required_keys:
        if key not in sample_item:
            raise AssertionError(f"Missing key {key} in dataset item")
        if not isinstance(sample_item[key], torch.Tensor):
            raise AssertionError(f"{key} is not a tensor")

    print("Dataset structure verified.")

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    model = SDFRNModel().to(device)

    # Verify Model Output Shape with a dummy batch
    dummy_batch = next(iter(train_loader))
    seq = dummy_batch["seq"].to(device)
    struct = dummy_batch["struct"].to(device)
    loop = dummy_batch["loop"].to(device)
    pid = dummy_batch["partner_id"].to(device)
    pidx = dummy_batch["partner_idx"].to(device)

    with torch.no_grad():
        # Pass 1: Initial prediction
        pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)
        # Pass 2: Feedback prediction
        pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)

    # Expected shape: (Batch, Seq_Len, 5)
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, 5)
    if pred2.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {pred2.shape}"
        )

    print(f"Model forward pass successful. Output shape: {pred2.shape}")

    # 4. Loss Function Verification
    print("\n--- Loss Verification ---")
    targets = dummy_batch["targets"].to(device)
    loss_val = mcrmse_loss(pred2, targets)

    if torch.isnan(loss_val):
        raise AssertionError("Loss returned NaN")
    if loss_val.item() < 0:
        raise AssertionError("Loss must be non-negative")

    print(f"MCRMSE Loss calculation successful: {loss_val.item():.4f}")

    # 5. Training Loop Demonstration
    print("\n--- Training Loop Demo ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    for epoch in range(Config.EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # Validate
        val_metric = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val MCRMSE = {val_metric:.4f}"
        )

        # Validation checks
        if train_loss <= 0:
            raise AssertionError("Train loss should be positive")
        if val_metric <= 0:
            raise AssertionError("Validation metric should be positive")

    # Save the model
    model_path = os.path.join(Config.CACHE_DIR, "demo_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 6. Inference and Submission
    print("\n--- Inference & Submission ---")
    # Load test data (mini subset)
    full_test_data = process_data("test", load_cached_data=False)
    mini_test_data = create_subset(full_test_data, size=8)

    test_dataset = RNADataset(mini_test_data, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            pid = batch["partner_id"].to(device)
            pidx = batch["partner_idx"].to(device)

            # Two-pass inference
            pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)
            pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)
            preds.append(pred2.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    if preds.shape != (8, Config.SEQ_LEN, 5):
        raise AssertionError(f"Prediction shape mismatch: {preds.shape}")

    # Generate Submission File
    submission_rows = []
    test_ids = mini_test_data["ids"]

    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    for i, sample_id in enumerate(test_ids):
        sample_pred = preds[i]
        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            vals = sample_pred[j]
            submission_rows.append([row_id] + vals.tolist())

    sub_df = pd.DataFrame(submission_rows, columns=["id_seqpos"] + Config.ALL_TARGETS)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {sub_df.shape}")

    # Validate submission content
    if sub_df.isnull().values.any():
        raise AssertionError("Submission contains NaNs")
    if sub_df.shape[1] != 6:
        raise AssertionError("Submission should have 6 columns")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
