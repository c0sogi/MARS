import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

# 1. Override Configuration for Fast Demo Execution
# We must do this before importing other modules that might rely on Config values during initialization,
# although most modules reference Config attributes dynamically.
from library.config import Config

print(">>> Configuring environment for Demo Run...")
Config.WORKING_DIR = "./working/demo_run"
Config.CACHE_DIR = Config.WORKING_DIR  # Isolate cache for this run
Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 molecules
Config.EPOCHS = 2
Config.BATCH_SIZE = 8
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
Config.HIDDEN_DIM = 32  # Smaller model for speed
Config.NUM_INTERACTION_LAYERS = 2

# Clean up any previous demo artifacts
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Import library modules after config setup
from library.utils import compute_bond_angle, TargetStandardizer, map_atom_types
from library.features import GaussianSmearing
from library.dataset import MoleculeDataset
from library.model import DualGraphNetwork
from library.loss import PhysicsAwareLoss
from library.trainer import Trainer


def verify_utils():
    print("\n>>> Verifying Utils...")

    # 1. Test Bond Angle Calculation
    # Construct 3 atoms: j at origin, i at (1,0,0), k at (0,1,0) -> 90 degrees -> cos=0
    pos_j = torch.tensor([[0.0, 0.0, 0.0]])
    pos_i = torch.tensor([[1.0, 0.0, 0.0]])
    pos_k = torch.tensor([[0.0, 1.0, 0.0]])

    cos_angle = compute_bond_angle(pos_i, pos_j, pos_k)
    print(f"  Computed cos(90 deg): {cos_angle.item():.4f}")
    assert (
        torch.abs(cos_angle - 0.0) < 1e-5
    ), "Bond angle calculation failed (expected 0.0)"

    # Test Parallel: i=(1,0,0), k=(2,0,0) -> 0 degrees -> cos=1
    pos_k_par = torch.tensor([[2.0, 0.0, 0.0]])
    cos_angle_par = compute_bond_angle(pos_i, pos_j, pos_k_par)
    print(f"  Computed cos(0 deg): {cos_angle_par.item():.4f}")
    assert (
        torch.abs(cos_angle_par - 1.0) < 1e-5
    ), "Bond angle calculation failed (expected 1.0)"

    # 2. Test Atom Mapping
    atoms = ["H", "C", "N", "O", "F", "X"]  # X is invalid
    indices = map_atom_types(atoms)
    print(f"  Atom Mapping {atoms} -> {indices}")
    expected = np.array([0, 1, 2, 3, 4, -1])
    assert np.array_equal(indices, expected), "Atom mapping failed"

    # 3. Test Target Standardizer
    print("  Testing TargetStandardizer...")
    std = TargetStandardizer()
    # Create dummy data: Type A has mean 10, Type B has mean 20
    df = pd.DataFrame(
        {
            "type": ["A", "A", "B", "B"],
            "scalar_coupling_constant": [9.0, 11.0, 19.0, 21.0],
        }
    )
    # Mock the config types for this test
    std.coupling_types = ["A", "B"]
    std.fit(df)

    assert std.stats["A"]["mean"] == 10.0
    assert std.stats["B"]["mean"] == 20.0

    transformed = std.transform(df)
    # 9.0 -> (9-10)/sqrt(2) approx -0.707
    # std dev of [9, 11] is sqrt(2) = 1.414
    expected_val = (9.0 - 10.0) / np.std([9.0, 11.0], ddof=1)
    assert (
        np.abs(transformed[0] - expected_val) < 1e-4
    ), "Standardization transform failed"

    # Restore
    restored = std.inverse_transform(transformed, df["type"].values)
    assert np.allclose(
        restored, df["scalar_coupling_constant"].values
    ), "Inverse transform failed"
    print("  Utils verification passed.")


def verify_features():
    print("\n>>> Verifying Features...")
    # Test Gaussian Smearing
    smear = GaussianSmearing(start=0, stop=10, num_gaussians=5)
    x = torch.tensor([0.0, 5.0, 10.0])
    out = smear(x)
    print(f"  Gaussian Smearing Output Shape: {out.shape}")
    assert out.shape == (3, 5), "Gaussian Smearing output shape incorrect"
    # At x=0, the first gaussian (centered at 0) should be high (1.0)
    assert out[0, 0] > 0.9, "Gaussian Smearing value incorrect at center"
    print("  Features verification passed.")


def verify_data_pipeline():
    print("\n>>> Verifying Data Pipeline (Dataset & Graph Builder)...")
    # Initialize Dataset (this triggers DualGraphBuilder)
    # We use load_cached=False to force processing of the debug subset
    dataset = MoleculeDataset(split="train", load_cached=False)

    print(f"  Dataset size (molecules): {len(dataset)}")
    assert len(dataset) > 0, "Dataset is empty"
    assert len(dataset) <= Config.DEBUG_SAMPLE_SIZE, "Dataset size exceeds debug limit"

    # Check one sample
    data = dataset[0]
    print(f"  Sample 0: {data}")

    # Verify Shapes
    assert data.x.dim() == 1, "Node features should be 1D (indices)"
    assert data.edge_index.shape[0] == 2, "Edge index should be (2, E)"
    assert (
        data.y.shape[0] == data.target_index.shape[1]
    ), "Number of targets does not match target indices"

    # Verify Batching Logic
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print(f"  Batch (size 2): {batch}")

    # Check if batch attributes exist
    assert hasattr(batch, "batch"), "Standard batch vector missing"
    assert hasattr(batch, "target_batch"), "Target batch vector missing"
    assert batch.num_graphs == 2, "Batch size incorrect"

    print("  Data pipeline verification passed.")
    return loader


def verify_model_and_loss(loader):
    print("\n>>> Verifying Model and Loss...")
    device = Config.DEVICE
    model = DualGraphNetwork().to(device)
    criterion = PhysicsAwareLoss()

    # Get a batch
    batch = next(iter(loader)).to(device)

    # Forward Pass
    preds = model(batch)
    pred_coupling, pred_shielding, pred_charges = preds

    print(f"  Pred Coupling Shape: {pred_coupling.shape}")
    print(f"  Pred Shielding Shape: {pred_shielding.shape}")

    # Assert Shapes
    assert pred_coupling.shape == (
        batch.y.shape[0],
        1,
    ), "Coupling prediction shape mismatch"
    assert pred_shielding.shape == (
        batch.x.shape[0],
        9,
    ), "Shielding prediction shape mismatch"
    assert pred_charges.shape == (
        batch.x.shape[0],
        1,
    ), "Charge prediction shape mismatch"

    # Loss Calculation
    loss, metrics = criterion(preds, batch)
    print(f"  Calculated Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")

    assert torch.isfinite(loss), "Loss is not finite"
    assert loss.item() > 0, "Loss should be positive"

    print("  Model and Loss verification passed.")


def verify_training_loop():
    print("\n>>> Verifying Full Training Loop...")

    # Instantiate Trainer
    trainer = Trainer()

    # 1. Train
    print("  Starting Trainer.train()...")
    trainer.train()

    # Check if best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved"
    print(f"  Model saved at {best_model_path}")

    # 2. Predict (Inference on Test)
    # Note: MoleculeDataset(split='test') will run builder on test set
    print("  Starting Trainer.predict()...")
    trainer.predict()

    # Check submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission generated with shape: {df_sub.shape}")
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission columns missing"

    print("  Training loop verification passed.")


if __name__ == "__main__":
    print("=== Starting Scalar Coupling Demo Script ===")

    try:
        verify_utils()
        verify_features()
        loader = verify_data_pipeline()
        verify_model_and_loss(loader)
        verify_training_loop()

        print("\n=== All Demonstrations Completed Successfully ===")

    except AssertionError as e:
        print(f"\n!!! VERIFICATION FAILED: {e} !!!")
        exit(1)
    except Exception as e:
        print(f"\n!!! RUNTIME ERROR: {e} !!!")
        import traceback

        traceback.print_exc()
        exit(1)
