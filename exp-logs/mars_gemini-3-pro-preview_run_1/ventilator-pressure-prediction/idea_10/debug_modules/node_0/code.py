import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.features import prepare_datasets
from library.dataset import get_dataloaders
from library.model import DeepSupervisedVentilatorModel
from library.loss import CompositeLoss, MaskedL1Loss
from library.train import train_epoch, validate_epoch
from library.inference import predict


def main():
    print("=== Starting Demonstration of Ventilator Pressure Prediction Library ===")

    # 1. Configure for Speed (Runtime Overrides)
    print("\n[1] Configuring environment for rapid execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small subset for demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean working directory for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Configuration: DEBUG={Config.DEBUG}, Device={device}")

    # 2. Data Preparation & Feature Engineering
    print("\n[2] Testing Data Preparation (library.features)...")
    # Force reload to demonstrate feature engineering pipeline
    data_tuple = prepare_datasets(load_cached_data=False)

    train_x, train_y, val_x, val_y, test_x, test_ids, center, scale = data_tuple

    # Verification
    expected_seq_len = Config.SEQ_LEN
    expected_features = Config.INPUT_DIM

    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")

    assert train_x.ndim == 3, "Train X should be 3D (N, Seq, Feat)"
    assert (
        train_x.shape[1] == expected_seq_len
    ), f"Sequence length mismatch. Expected {expected_seq_len}"
    assert (
        train_x.shape[2] == expected_features
    ), f"Feature dimension mismatch. Expected {expected_features}"
    assert train_y.shape == (
        train_x.shape[0],
        expected_seq_len,
    ), "Target shape mismatch"
    assert (
        len(test_ids) == test_x.shape[0] * expected_seq_len
    ), "Test IDs should match flattened test data size"

    print("Data preparation verification passed.")

    # 3. DataLoaders
    print("\n[3] Testing DataLoaders (library.dataset)...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,  # Should hit the cache we just created
    )

    # Fetch one batch
    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    print(f"Batch X shape: {batch_x.shape}")
    print(f"Batch Y shape: {batch_y.shape}")

    assert batch_x.shape[0] == Config.BATCH_SIZE
    assert batch_x.dtype == torch.float32

    print("DataLoader verification passed.")

    # 4. Model Architecture
    print("\n[4] Testing Model Architecture (library.model)...")
    model = DeepSupervisedVentilatorModel().to(device)

    # Test Training Mode (Deep Supervision: returns tuple)
    model.train()
    outputs = model(batch_x)
    assert isinstance(
        outputs, tuple
    ), "Model in train mode should return tuple (final, aux)"
    final_pred, aux_pred = outputs

    assert (
        final_pred.shape == batch_y.shape
    ), f"Final prediction shape mismatch: {final_pred.shape}"
    assert (
        aux_pred.shape == batch_y.shape
    ), f"Aux prediction shape mismatch: {aux_pred.shape}"

    # Test Eval Mode (Inference: returns tensor)
    model.eval()
    with torch.no_grad():
        inference_pred = model(batch_x)
    assert isinstance(
        inference_pred, torch.Tensor
    ), "Model in eval mode should return Tensor"
    assert inference_pred.shape == batch_y.shape

    print("Model architecture verification passed.")

    # 5. Loss Functions
    print("\n[5] Testing Loss Functions (library.loss)...")
    # Extract u_out from batch (index 2 in feature cols)
    u_out_idx = Config.FEATURE_COLS.index("u_out")
    u_out = batch_x[:, :, u_out_idx]

    # Composite Loss (Train)
    train_criterion = CompositeLoss().to(device)
    loss_val = train_criterion((final_pred, aux_pred), batch_y, u_out)
    print(f"Composite Loss Value: {loss_val.item()}")
    assert not torch.isnan(loss_val), "Composite loss is NaN"
    assert loss_val > 0, "Loss should be positive"

    # Masked L1 Loss (Val)
    val_criterion = MaskedL1Loss().to(device)
    mae_val = val_criterion(inference_pred, batch_y, u_out)
    print(f"Masked MAE Value: {mae_val.item()}")
    assert not torch.isnan(mae_val), "MAE loss is NaN"

    print("Loss function verification passed.")

    # 6. Training Loop Execution
    print("\n[6] Testing Training Loop (library.train)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
    )

    # Run 1 Epoch
    print("Running training epoch...")
    train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, train_criterion, device, None
    )
    print(f"Train Loss: {train_loss:.4f}")

    print("Running validation epoch...")
    val_mae = validate_epoch(model, val_loader, val_criterion, device)
    print(f"Val MAE: {val_mae:.4f}")

    assert train_loss > 0
    assert val_mae > 0

    # Save model for inference step
    torch.save(model.state_dict(), Config.MODEL_PATH)
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved"

    print("Training loop verification passed.")

    # 7. Inference Pipeline
    print("\n[7] Testing Inference Pipeline (library.inference)...")
    # predict() loads the model from Config.MODEL_PATH and generates submission
    submission_df = predict(load_cached_data=True)

    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())

    # Verify Submission
    expected_rows = test_x.shape[0] * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Submission rows mismatch. Expected {expected_rows}, got {len(submission_df)}"
    assert list(submission_df.columns) == [
        "id",
        "pressure",
    ], "Submission columns mismatch"
    assert submission_df["id"].is_monotonic_increasing, "IDs should be sorted/monotonic"

    print("Inference pipeline verification passed.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
