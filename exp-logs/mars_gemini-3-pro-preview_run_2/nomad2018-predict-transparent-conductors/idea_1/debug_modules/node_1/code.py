import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.utils import set_seed, Normalizer, AverageMeter
from library.data_loader import get_train_val_test_loaders
from library.model import CGCNN
from library.trainer import Trainer


def test_utils():
    print("--- Testing Utils ---")
    set_seed(42)

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Total sum = 10*2 + 20*2 = 60. Total count = 4. Avg = 15.
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    print("AverageMeter verified.")

    # Test Normalizer
    data = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    normalizer = Normalizer(tensor=data)

    # Check mean and std
    expected_mean = torch.tensor([2.0, 20.0])
    expected_std = torch.tensor([1.0, 10.0])  # std of [1,2,3] is 1.0

    assert torch.allclose(
        normalizer.mean, expected_mean
    ), "Normalizer mean calculation failed"
    assert torch.allclose(
        normalizer.std, expected_std
    ), "Normalizer std calculation failed"

    # Test normalization and denormalization
    normed = normalizer.norm(data)
    denormed = normalizer.denorm(normed)
    assert torch.allclose(data, denormed, atol=1e-6), "Normalizer reconstruction failed"

    # Test state dict
    state = normalizer.state_dict()
    new_normalizer = Normalizer()
    new_normalizer.load_state_dict(state)
    assert torch.allclose(
        new_normalizer.mean, normalizer.mean
    ), "Normalizer state loading failed"
    print("Normalizer verified.")


def test_data_loader():
    print("\n--- Testing Data Loader ---")
    # Use a small radius to speed up graph construction if processing is needed
    # (though it likely loads from cache or processes quickly)
    radius = 4.0
    batch_size = 4

    # Note: The metadata files are expected to be in ./metadata as per problem description
    # and the logic in library/data_loader.py

    # We use num_workers=0 to avoid multiprocessing overhead in this demonstration
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        batch_size=batch_size,
        radius=radius,
        num_workers=0,
        load_cached_data=False,  # Force processing to verify logic, or True if cache exists
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    atom_fea, edge_index, edge_dist, batch_index, targets, ids = batch

    print(f"Batch keys shapes:")
    print(f"  Atom features: {atom_fea.shape}")  # [N_nodes]
    print(f"  Edge index:    {edge_index.shape}")  # [2, N_edges]
    print(f"  Edge dist:     {edge_dist.shape}")  # [N_edges]
    print(f"  Batch index:   {batch_index.shape}")  # [N_nodes]
    print(f"  Targets:       {targets.shape}")  # [Batch_size, 2]
    print(f"  IDs:           {ids.shape}")  # [Batch_size]

    assert atom_fea.dim() == 1, "Atom features should be 1D (indices)"
    assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert targets.shape[1] == 2, "Targets should have 2 columns (formation, bandgap)"
    assert len(ids) == batch_size, "Batch size mismatch for IDs"

    print("Data Loader verified.")
    return train_loader, val_loader, test_loader


def test_model_and_training(train_loader, val_loader, test_loader):
    print("\n--- Testing Model and Training Loop ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Hyperparameters
    orig_atom_fea_len = 4  # Al, Ga, In, O
    atom_fea_len = 32
    n_conv = 2
    h_fea_len = 32
    n_h = 1
    n_targets = 2
    radius = 4.0  # Must match data loader
    n_rbf = 20

    model = CGCNN(
        orig_atom_fea_len=orig_atom_fea_len,
        atom_fea_len=atom_fea_len,
        n_conv=n_conv,
        h_fea_len=h_fea_len,
        n_h=n_h,
        n_targets=n_targets,
        radius=radius,
        n_rbf=n_rbf,
    ).to(device)

    # Verify forward pass with a single batch
    batch = next(iter(train_loader))
    atom_fea, edge_index, edge_dist, batch_index, targets, ids = [
        x.to(device) for x in batch
    ]

    output = model(atom_fea, edge_index, edge_dist, batch_index)
    assert (
        output.shape == targets.shape
    ), f"Model output shape mismatch. Expected {targets.shape}, got {output.shape}"
    print("Model forward pass verified.")

    # Setup Trainer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize normalizer from training data statistics
    # For speed, we just use the first batch to set mean/std in this demo
    normalizer = Normalizer(tensor=targets)
    normalizer.mean = normalizer.mean.to(device)
    normalizer.std = normalizer.std.to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        normalizer=normalizer,
    )

    # Run one epoch of training
    print("Running 1 epoch of training...")
    train_loss = trainer.train_epoch(train_loader, epoch=1)
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run validation
    print("Running validation...")
    val_loss, val_rmsle = trainer.validate(val_loader)
    print(f"Val Loss: {val_loss:.4f}, Val RMSLE: {val_rmsle:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # Run prediction
    print("Running prediction on test set...")
    output_csv = "./working/submission_demo.csv"
    trainer.predict(test_loader, output_csv)

    assert os.path.exists(output_csv), "Output CSV not created"
    df = pd.read_csv(output_csv)
    assert len(df) == len(test_loader.dataset), "Output CSV length mismatch"
    assert "formation_energy_ev_natom" in df.columns, "Missing formation energy column"
    assert "bandgap_energy_ev" in df.columns, "Missing bandgap energy column"

    print("Training and Prediction verified.")


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs("./working", exist_ok=True)

    # 1. Verify Utilities
    test_utils()

    # 2. Verify Data Loading
    # We use a smaller batch size for the demo to ensure we see multiple batches even with small data
    train_loader, val_loader, test_loader = test_data_loader()

    # 3. Verify Model and Trainer
    test_model_and_training(train_loader, val_loader, test_loader)

    print("\nAll demonstrations completed successfully.")
