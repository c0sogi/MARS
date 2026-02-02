import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import process_data, RNADataset
from library.model import EIPFN
from library.loss_metric import MCRMSELoss
from library.engine import train_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_dataloaders(batch_size=4):
    """
    Creates DataLoaders using a small subset of the metadata to ensure speed.
    """
    print("\n[Demo] Creating mini datasets...")

    # Load subsets of metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv")).head(32)
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv")).head(16)
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv")).head(16)

    # Process data (convert to numpy arrays/features)
    # We bypass the caching logic in get_loaders for this quick demo
    train_dict = process_data(train_df, mode="train")
    val_dict = process_data(val_df, mode="val")
    test_dict = process_data(test_df, mode="test")

    # Create Datasets
    train_dataset = RNADataset(train_dict, mode="train")
    val_dataset = RNADataset(val_dict, mode="val")
    test_dataset = RNADataset(test_dict, mode="test")

    # Create Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"[Demo] Mini Train size: {len(train_dataset)}")
    print(f"[Demo] Mini Val size: {len(val_dataset)}")
    print(f"[Demo] Mini Test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, test_df


def verify_model_architecture(device):
    """
    Instantiates the model and verifies output shapes with dummy data.
    """
    print("\n[Demo] Verifying Model Architecture...")
    model = EIPFN().to(device)

    # Create dummy input
    # Shape: (Batch, Channels, SeqLen)
    B, C, L = 2, Config.INPUT_CHANNELS, Config.SEQ_LEN
    dummy_input = torch.randn(B, C, L).to(device)

    # Create dummy partner map (indices 0..L-1)
    dummy_pmap = torch.arange(L).unsqueeze(0).repeat(B, 1).to(device)

    # Forward pass
    y1, y2 = model(dummy_input, dummy_pmap)

    # Check shapes
    expected_shape = (B, Config.NUM_TARGETS, L)

    assert (
        y1.shape == expected_shape
    ), f"Pass 1 output shape mismatch. Expected {expected_shape}, got {y1.shape}"
    assert (
        y2.shape == expected_shape
    ), f"Pass 2 output shape mismatch. Expected {expected_shape}, got {y2.shape}"

    print("[Demo] Model architecture verified successfully.")
    return model


def verify_loss_metric(device):
    """
    Verifies the MCRMSELoss logic against a manual calculation.
    """
    print("\n[Demo] Verifying Loss Logic...")
    criterion = MCRMSELoss()

    # Create dummy predictions and targets
    # Shape: (Batch=1, Targets=5, SeqLen=107)
    # Scored targets are indices [0, 1, 3]
    # Scored length is 68

    y_pred = torch.zeros(1, 5, 107).to(device)
    y_true = torch.zeros(1, 5, 107).to(device)

    # Set specific errors for scored columns within scored length
    # Col 0 (reactivity): error 1.0 -> MSE=1.0 -> RMSE=1.0
    y_pred[0, 0, 0:68] = 1.0
    y_true[0, 0, 0:68] = 0.0

    # Col 1 (deg_Mg_pH10): error 2.0 -> MSE=4.0 -> RMSE=2.0
    y_pred[0, 1, 0:68] = 2.0
    y_true[0, 1, 0:68] = 0.0

    # Col 3 (deg_Mg_50C): error 0.0 -> MSE=0.0 -> RMSE=0.0
    y_pred[0, 3, 0:68] = 0.0
    y_true[0, 3, 0:68] = 0.0

    # Set errors outside scored length (should be ignored)
    y_pred[0, 0, 70] = 100.0

    # Set errors in unscored columns (should be ignored)
    y_pred[0, 2, 0] = 100.0

    # Expected MCRMSE = mean([1.0, 2.0, 0.0]) = 1.0
    loss = criterion(y_pred, y_true)

    assert (
        abs(loss.item() - 1.0) < 1e-5
    ), f"Loss verification failed. Expected 1.0, got {loss.item()}"

    print("[Demo] Loss logic verified successfully.")


def run_inference(model, test_loader, test_df, device):
    """
    Runs inference and generates a submission file.
    """
    print("\n[Demo] Running Inference...")
    model.eval()

    preds = []
    ids = []

    with torch.no_grad():
        for inputs, partner_map, sample_ids in test_loader:
            inputs = inputs.to(device)
            partner_map = partner_map.to(device)

            # Use Pass 2 predictions
            _, y_pred = model(inputs, partner_map)

            # Move to CPU
            y_pred = y_pred.cpu().numpy()

            preds.append(y_pred)
            ids.extend(sample_ids)

    preds = np.concatenate(preds, axis=0)

    # Generate Submission DataFrame
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    print("[Demo] Formatting submission...")
    for i, sample_id in enumerate(ids):
        # Retrieve sequence length for this sample from dataframe
        # (Though in this dataset all are 107, we handle it generically)
        seq_len = test_df[test_df["id"] == sample_id]["seq_length"].values[0]

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"
            row_data = {"id_seqpos": row_id}

            for t_idx, col in enumerate(target_cols):
                # preds shape: (N, 5, 107)
                val = preds[i, t_idx, pos]
                row_data[col] = float(val)

            submission_rows.append(row_data)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    out_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"[Demo] Submission saved to {out_path}")
    print(f"[Demo] Submission shape: {submission_df.shape}")

    # Verify submission content
    assert not submission_df.isnull().values.any(), "Submission contains NaNs"
    assert "id_seqpos" in submission_df.columns
    assert len(submission_df) == len(test_df) * 107


if __name__ == "__main__":
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"[Demo] Running on device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader, test_df = create_mini_dataloaders()

    # 3. Verification
    model = verify_model_architecture(device)
    verify_loss_metric(device)

    # 4. Training
    print("\n[Demo] Starting Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    # Train for 2 epochs only for demo
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=2,
        device=device,
        patience=2,
    )

    # 5. Inference
    # Load best model (saved by train_model)
    print(f"\n[Demo] Loading best model from {Config.MODEL_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    run_inference(model, test_loader, test_df, device)

    print("\n[Demo] Execution Complete.")
