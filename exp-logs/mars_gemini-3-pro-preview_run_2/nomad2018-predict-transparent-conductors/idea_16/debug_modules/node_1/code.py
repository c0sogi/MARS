import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    compute_rmsle,
    count_parameters,
)
from library.data import (
    GaussianDistance,
    StandardScaler,
    CrystalGraphDataset,
    collate_graphs,
)
from library.model import IR_CGCNN
from library.train import train_one_epoch, validate


def run_demo():
    print("Starting IR-CGCNN Pipeline Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for a quick demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Use a tiny subset and minimal epochs for speed
    Config.DEBUG_SAMPLE_SIZE = 50
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Use main process for simplicity in demo

    # Ensure directories exist (since we changed paths)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Processing Components
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Components...")

    # Test GaussianDistance
    gd = GaussianDistance(dmin=0, dmax=5, step=1.0)
    distances = np.array([0.0, 2.5, 5.0])
    expanded = gd.expand(distances)
    print(f"    GaussianDistance output shape: {expanded.shape}")
    assert expanded.shape == (3, 6), "GaussianDistance expansion shape mismatch"

    # Test StandardScaler
    scaler = StandardScaler()
    dummy_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler.fit(dummy_data)
    transformed = scaler.transform(dummy_data)
    inverse = scaler.inverse_transform(transformed)
    print("    StandardScaler fit/transform/inverse check passed.")
    assert np.allclose(dummy_data, inverse), "StandardScaler inverse transform failed"

    # -------------------------------------------------------------------------
    # 3. Dataset & Dataloader
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Dataset and Dataloader...")

    # Initialize Dataset (this will process graphs and save to cache)
    # We use 'train' mode which expects targets
    train_dataset = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        mode="train",
        load_cached_data=False,  # Force processing for demo purposes
    )

    print(f"    Dataset length: {len(train_dataset)}")
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), "Dataset did not respect DEBUG_SAMPLE_SIZE"

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    atom_fea, edge_index, edge_fea, batch_index, targets, ids = batch

    print(f"    Batch Atom Features Shape: {atom_fea.shape}")
    print(f"    Batch Edge Index Shape: {edge_index.shape}")
    print(f"    Batch Targets Shape: {targets.shape}")

    # Move batch to device for model testing
    atom_fea = atom_fea.to(device)
    edge_index = edge_index.to(device)
    edge_fea = edge_fea.to(device)
    batch_index = batch_index.to(device)
    targets = targets.to(device)

    # -------------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Model Instantiation & Forward Pass...")

    model = IR_CGCNN(Config).to(device)
    num_params = count_parameters(model)
    print(f"    Model created with {num_params} trainable parameters.")

    # Run forward pass
    preds = model(atom_fea, edge_index, edge_fea, batch_index)
    print(f"    Prediction Shape: {preds.shape}")

    assert preds.shape == (len(ids), 2), "Prediction shape mismatch (Batch_Size, 2)"
    assert not torch.isnan(preds).any(), "Model produced NaN predictions"

    # -------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[5] Simulating Training Loop...")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run for a few epochs
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"    Epoch {epoch}: Train Loss (MSE) = {loss:.6f}")
        assert not np.isnan(loss), "Training loss is NaN"

    # Validate
    # We reuse train_loader as val_loader for this demo
    val_loss, val_rmsle = validate(
        model, train_loader, criterion, device, train_dataset.scaler
    )
    print(f"    Validation Loss: {val_loss:.6f}")
    print(f"    Validation RMSLE: {val_rmsle:.6f}")

    # -------------------------------------------------------------------------
    # 6. Checkpointing & Utilities
    # -------------------------------------------------------------------------
    print("\n[6] Testing Checkpointing & Metrics...")

    # Save Checkpoint
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "demo_model.pth")
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": Config.NUM_EPOCHS,
            "val_rmsle": val_rmsle,
        },
        ckpt_path,
    )
    print(f"    Checkpoint saved to {ckpt_path}")

    # Load Checkpoint
    loaded_ckpt = load_checkpoint(ckpt_path, model, optimizer, device=Config.DEVICE)
    print(f"    Checkpoint loaded. Epoch: {loaded_ckpt['epoch']}")

    # Test RMSLE computation
    y_true = np.array([[0.1, 1.5], [0.2, 2.0]])
    y_pred = np.array([[0.12, 1.4], [0.22, 2.1]])
    metric = compute_rmsle(y_pred, y_true)
    print(f"    Test RMSLE calculation: {metric:.6f}")
    assert metric >= 0, "RMSLE should be non-negative"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
