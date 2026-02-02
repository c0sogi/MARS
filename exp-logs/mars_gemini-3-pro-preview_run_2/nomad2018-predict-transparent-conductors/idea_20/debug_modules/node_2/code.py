import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Force reload of library modules to ensure changes are picked up in persistent environment
for key in list(sys.modules.keys()):
    if key.startswith("library"):
        del sys.modules[key]

from library.utils import set_seed, compute_rmsle, StandardScaler
from library.data import get_dataloaders, GaussianSmearing, process_structure
from library.model import AICGN
from library.train import Trainer


def test_utils():
    print("Testing Utils...")
    set_seed(42)

    # Test StandardScaler
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    scaler.fit(data)

    transformed = scaler.transform(data)
    expected_mean = torch.tensor([0.0, 0.0])
    # Std of [1, 3, 5] is 2.0. (1-3)/2 = -1, (3-3)/2 = 0, (5-3)/2 = 1
    assert torch.allclose(
        transformed.mean(dim=0), expected_mean, atol=1e-6
    ), "Scaler mean failed"

    inverse = scaler.inverse_transform(transformed)
    assert torch.allclose(inverse, data, atol=1e-6), "Scaler inverse failed"
    print("  StandardScaler passed.")

    # Test RMSLE
    y_true = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    y_pred = torch.tensor([[1.1, 9.5], [1.9, 21.0]])
    # Just ensure it runs and returns a float
    loss = compute_rmsle(y_pred, y_true)
    assert isinstance(loss, float), "RMSLE should return a float"
    print(f"  RMSLE check passed (Loss: {loss:.4f}).")


def test_data_processing():
    print("\nTesting Data Processing...")

    # Test Gaussian Smearing
    smearing = GaussianSmearing(0.0, 5.0, 10)
    distances = torch.tensor([1.0, 2.5, 4.0])
    features = smearing(distances)
    assert features.shape == (
        3,
        10,
    ), f"Smearing output shape mismatch: {features.shape}"
    print("  GaussianSmearing passed.")

    # Test process_structure with a real file from input
    # Assuming standard structure: input/train/1/geometry.xyz
    sample_path = "train/1/geometry.xyz"
    if os.path.exists(os.path.join("./input", sample_path)):
        data = process_structure(sample_path, targets=[0.5, 1.5], cutoff=5.0)
        assert data.x is not None, "Data object missing node features"
        assert data.edge_index is not None, "Data object missing edge index"
        assert data.edge_attr is not None, "Data object missing edge attributes"
        assert data.y is not None, "Data object missing targets"
        print(
            f"  Single structure processing passed ({data.num_nodes} nodes, {data.num_edges} edges)."
        )
    else:
        print("  Skipping structure test (sample file not found).")


def test_dataset_and_loader():
    print("\nTesting Dataset and DataLoader...")

    # Use a small sample size and disable cache loading to force processing
    batch_size = 4
    sample_size = 20

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,
        load_cached_data=False,
        sample_size=sample_size,
    )

    # Check train loader
    batch = next(iter(train_loader))
    print(f"  Train Batch: {batch}")
    assert batch.x.ndim == 1, "Node features should be 1D (atomic numbers)"
    assert batch.edge_attr.ndim == 2, "Edge features should be 2D"
    assert batch.y.shape[1] == 2, "Targets should have 2 columns"
    assert batch.batch is not None, "Batch vector missing"

    print("  DataLoaders initialized successfully.")
    return train_loader, val_loader, test_loader


def test_model_architecture(device):
    print("\nTesting Model Architecture...")

    model = AICGN(
        node_input_dim=100, edge_input_dim=60, hidden_dim=32, num_layers=2, dropout=0.0
    ).to(device)

    # Create dummy data
    # 10 nodes, 20 edges
    x = torch.randint(0, 100, (10,)).to(device)
    edge_index = torch.randint(0, 10, (2, 20)).to(device)
    edge_attr = torch.randn(20, 60).to(device)
    batch = torch.zeros(10, dtype=torch.long).to(device)

    from torch_geometric.data import Data

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    out_form, out_band = model(data)

    assert out_form.shape == (
        1,
        1,
    ), f"Formation output shape mismatch: {out_form.shape}"
    assert out_band.shape == (1, 1), f"Bandgap output shape mismatch: {out_band.shape}"

    print("  Model forward pass successful.")
    return model


def test_training_pipeline(model, train_loader, val_loader, test_loader, device):
    print("\nTesting Training Pipeline...")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=2,
        checkpoint_dir="./working/test_checkpoints",
    )

    # Run training for a few epochs
    print("  Running fit (2 epochs)...")
    trainer.fit(train_loader, val_loader, epochs=2)

    # Test Prediction
    print("  Running prediction on test set...")
    ids, preds = trainer.predict(test_loader)

    assert len(ids) == len(test_loader.dataset), "Prediction count mismatch"
    assert preds.shape == (len(ids), 2), "Prediction shape mismatch"

    print("  Training and Inference pipeline passed.")
    print(
        f"  Sample Predictions:\nID: {ids[0]}, Form: {preds[0][0]:.4f}, Band: {preds[0][1]:.4f}"
    )

    # Generate submission file
    print("\nGenerating Submission File...")
    output_path = "./working/submission_test.csv"

    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )
    df.to_csv(output_path, index=False)
    print(f"  Submission saved to {output_path}")
    assert os.path.exists(output_path), "Submission file was not created"


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    try:
        # 1. Test Utilities
        test_utils()

        # 2. Test Data Processing
        test_data_processing()

        # 3. Test Data Loading (Small subset)
        train_loader, val_loader, test_loader = test_dataset_and_loader()

        # 4. Test Model
        model = test_model_architecture(device)

        # 5. Test Training and Inference
        test_training_pipeline(model, train_loader, val_loader, test_loader, device)

        print("\nAll tests passed successfully!")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
