import os
import torch
import pandas as pd
import numpy as np
import library.config as config
import library.utils as utils
import library.features as features
import library.dataset as dataset
import library.model as model_lib
import library.loss as loss_lib
import library.engine as engine


def run_demo():
    print("=== Starting Ventilator Pressure Prediction Demo ===")

    # 1. Configuration Overrides for Speed and Isolation
    # We use DEBUG mode and a tiny sample size to ensure execution < 1 min (after data loading)
    config.SEED = 42
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 100  # 100 breaths per split
    config.EPOCHS = 1  # Single epoch
    config.BATCH_SIZE = 16  # Small batch size
    config.NUM_WORKERS = 0  # Run in main process to avoid overhead

    # Define a separate working directory for this demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Update config paths
    config.WORKING_DIR = demo_dir
    config.TRAIN_CACHE_PATH = os.path.join(demo_dir, "train_engineered.parquet")
    config.VAL_CACHE_PATH = os.path.join(demo_dir, "val_engineered.parquet")
    config.TEST_CACHE_PATH = os.path.join(demo_dir, "test_engineered.parquet")
    config.SCALER_CENTER_PATH = os.path.join(demo_dir, "scaler_center.npy")
    config.SCALER_SCALE_PATH = os.path.join(demo_dir, "scaler_scale.npy")
    config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model.pth")
    config.SUBMISSION_FILE_PATH = os.path.join(demo_dir, "submission.csv")

    # Patch modules that imported paths directly from config
    features.TRAIN_CACHE_PATH = config.TRAIN_CACHE_PATH
    features.VAL_CACHE_PATH = config.VAL_CACHE_PATH
    features.TEST_CACHE_PATH = config.TEST_CACHE_PATH
    features.SCALER_CENTER_PATH = config.SCALER_CENTER_PATH
    features.SCALER_SCALE_PATH = config.SCALER_SCALE_PATH
    model_lib.MODEL_SAVE_PATH = config.MODEL_SAVE_PATH

    # Set seeds
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading and Engineering
    print("\n--- Loading and Engineering Data ---")
    # This will load metadata, engineer features, scale, and then sample for DEBUG mode
    train_ds, val_ds, test_ds = dataset.get_ventilator_datasets(load_cached_data=True)

    # Verification: Dataset Shapes
    print("Verifying dataset integrity...")
    assert (
        len(train_ds) == config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(train_ds)}"

    sample_item = train_ds[0]
    seq_len = config.SEQ_LEN
    n_features = len(config.FEATURE_NAMES)

    # Check Tensor Shapes: (80, N_features)
    assert sample_item["x"].shape == (
        seq_len,
        n_features,
    ), f"Expected input shape ({seq_len}, {n_features}), got {sample_item['x'].shape}"
    assert sample_item["u_out"].shape == (seq_len,), "u_out shape mismatch"
    assert sample_item["y"].shape == (seq_len,), "Target shape mismatch"

    print("Dataset verification passed.")

    # 3. Model Architecture Verification
    print("\n--- Initializing and Verifying Model ---")
    model = model_lib.VentilatorModel().to(device)

    # Create dummy input batch: (Batch=2, Seq=80, Feats=N)
    dummy_x = torch.randn(2, seq_len, n_features).to(device)

    # Forward pass
    with torch.no_grad():
        pred, aux_pred = model(dummy_x)

    # Check output shapes: (Batch, Seq, 1)
    assert pred.shape == (2, seq_len, 1), f"Prediction shape error: {pred.shape}"
    assert aux_pred.shape == (
        2,
        seq_len,
        1,
    ), f"Aux prediction shape error: {aux_pred.shape}"
    print("Model architecture verification passed.")

    # 4. Loss Function Verification
    print("\n--- Verifying Custom Loss Function ---")
    loss_fn = loss_lib.MaskedL1Loss()

    # Scenario:
    # Batch=1, Seq=4.
    # u_out = [0, 0, 1, 1] (First 2 are inspiratory/valid, last 2 are expiratory/masked)
    # Target = [10, 10, 10, 10]
    # Pred   = [12, 12, 50, 50]
    # Errors:
    #   Idx 0: |12-10| = 2 (Valid)
    #   Idx 1: |12-10| = 2 (Valid)
    #   Idx 2: |50-10| = 40 (Masked -> 0)
    #   Idx 3: |50-10| = 40 (Masked -> 0)
    # Mean Error = (2 + 2) / 2 = 2.0

    t_u_out = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32).to(device)
    t_target = torch.tensor([[10, 10, 10, 10]], dtype=torch.float32).to(device)
    t_pred = (
        torch.tensor([[12, 12, 50, 50]], dtype=torch.float32).unsqueeze(-1).to(device)
    )

    calc_loss = loss_fn(t_pred, t_target, t_u_out, aux_pred=None)
    expected_loss = 2.0

    assert torch.isclose(
        calc_loss, torch.tensor(expected_loss).to(device)
    ), f"Loss mismatch. Expected {expected_loss}, got {calc_loss.item()}"
    print("Loss function verification passed.")

    # 5. Training Loop Execution
    print("\n--- Executing Training Loop (1 Epoch) ---")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
    )

    # Train
    train_loss = engine.train_one_epoch(
        model, train_loader, optimizer, scheduler, device, loss_fn
    )
    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN"

    # Evaluate
    val_mae = engine.evaluate(model, val_loader, device, loss_fn)
    print(f"Validation MAE: {val_mae:.4f}")
    assert not np.isnan(val_mae), "Validation MAE resulted in NaN"

    # 6. Inference and Submission
    print("\n--- Executing Inference ---")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    ids, preds = engine.predict(model, test_loader, device)

    # Verify inference output size
    expected_len = len(test_ds) * config.SEQ_LEN
    assert len(ids) == expected_len, f"ID count mismatch: {len(ids)} vs {expected_len}"
    assert (
        len(preds) == expected_len
    ), f"Prediction count mismatch: {len(preds)} vs {expected_len}"

    # Save Submission
    submission = pd.DataFrame({"id": ids, "pressure": preds})
    submission.to_csv(config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
    print("Top 5 rows:")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
