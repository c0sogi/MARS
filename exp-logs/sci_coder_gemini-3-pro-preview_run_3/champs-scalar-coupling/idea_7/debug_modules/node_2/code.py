import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import GaussianSmearing, TargetStandardizer
from library.dataset import MolecularGraphDataset
from library.model import SGLGN
from library.engine import run_training
from torch_geometric.loader import DataLoader


def demo_utils():
    print("\n=== Demonstrating Utils ===")

    # 1. Test GaussianSmearing
    print("Testing GaussianSmearing...")
    start, stop, n_gaussians = 0.0, 5.0, 10
    smearing = GaussianSmearing(start=start, stop=stop, num_gaussians=n_gaussians)

    # Create dummy distances: start, mid, stop
    dist = torch.tensor([0.0, 2.5, 5.0], dtype=torch.float)
    features = smearing(dist)

    assert features.shape == (
        3,
        n_gaussians,
    ), f"Expected shape (3, {n_gaussians}), got {features.shape}"
    assert features.max() <= 1.0 + 1e-6, "RBF values should be <= 1.0"
    print("GaussianSmearing check passed.")

    # 2. Test TargetStandardizer
    print("Testing TargetStandardizer...")
    standardizer = TargetStandardizer()

    # Create dummy dataframe
    data = {
        "type": ["1JHC", "1JHC", "2JHH", "2JHH"],
        "scalar_coupling_constant": [100.0, 110.0, -10.0, -12.0],
    }
    df = pd.DataFrame(data)

    # Fit
    standardizer.fit(df)
    stats = standardizer.stats
    assert "1JHC" in stats and "2JHH" in stats, "Stats missing coupling types"

    # Transform
    df_transformed = standardizer.transform(df)
    vals_1jhc = df_transformed[df_transformed["type"] == "1JHC"][
        "scalar_coupling_constant"
    ]
    assert abs(vals_1jhc.mean()) < 1e-5, "Standardized mean should be approx 0"

    # Inverse Transform (Manual Check with tensor)
    # Simulate predictions (z-scores)
    pred_z = torch.tensor([0.0, 0.0], dtype=torch.float)  # Mean prediction
    types = ["1JHC", "2JHH"]

    # We need to move tensors to Config.DEVICE if standardizer uses it,
    # but here we test the CPU/Numpy fallback or explicit device handling
    pred_orig = standardizer.inverse_transform(pred_z.to(Config.DEVICE), types)

    expected_1jhc = stats["1JHC"]["mean"]
    expected_2jhh = stats["2JHH"]["mean"]

    assert np.isclose(
        pred_orig[0].item(), expected_1jhc
    ), "Inverse transform failed for 1JHC"
    assert np.isclose(
        pred_orig[1].item(), expected_2jhh
    ), "Inverse transform failed for 2JHH"
    print("TargetStandardizer check passed.")


def demo_dataset_and_model():
    print("\n=== Demonstrating Dataset and Model ===")

    # Initialize Dataset (Train mode)
    # This will trigger the 'process' method which uses the reduced DEBUG_SAMPLE_SIZE
    print("Initializing MolecularGraphDataset (Train)...")
    dataset = MolecularGraphDataset(mode="train", load_cached_data=False)

    assert len(dataset) > 0, "Dataset is empty"
    print(f"Dataset size: {len(dataset)} graphs")

    # Inspect one graph
    data = dataset[0]
    print(f"Sample Graph Keys: {data.keys()}")

    # Verify essential attributes
    assert hasattr(data, "x"), "Missing node features (x)"
    assert hasattr(data, "edge_index"), "Missing edge_index"
    assert hasattr(data, "edge_attr"), "Missing edge_attr (distance RBF)"
    assert hasattr(data, "line_edge_index"), "Missing line_edge_index"
    assert hasattr(data, "target_edge_index"), "Missing target_edge_index"

    # Initialize Model
    print("Initializing SGLGN Model...")
    model = SGLGN().to(Config.DEVICE)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader)).to(Config.DEVICE)

    # Forward Pass
    print("Running Forward Pass...")
    model.eval()
    with torch.no_grad():
        preds, pred_shield, pred_charge = model(batch)

    # Verify Output Shapes
    num_targets = batch.target_type.size(0)
    num_atoms = batch.x.size(0)

    assert preds.shape == (
        num_targets,
    ), f"Preds shape mismatch: {preds.shape} vs ({num_targets},)"
    assert pred_shield.shape == (
        num_atoms,
        9,
    ), f"Shield shape mismatch: {pred_shield.shape}"
    assert pred_charge.shape == (
        num_atoms,
    ), f"Charge shape mismatch: {pred_charge.shape}"

    print("Model Forward Pass check passed.")


def demo_full_pipeline():
    print("\n=== Demonstrating Full Training Pipeline ===")
    print("Running Trainer.fit() and generate_submission()...")

    # This calls the engine's run_training which handles the loop
    run_training()

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
        assert "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
        print("Pipeline check passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # ==========================================
    # 1. Configure for Fast Demo
    # ==========================================
    print("Configuring environment for demo...")

    # Set Paths to a demo working directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.PROCESSED_DIR = os.path.join(Config.WORKING_DIR, "processed")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set Hyperparameters for Speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 molecules
    Config.EPOCHS = 1  # Train for 1 epoch
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set Seed
    Config.set_seed(42)

    # ==========================================
    # 2. Run Demonstrations
    # ==========================================
    try:
        demo_utils()
        demo_dataset_and_model()
        demo_full_pipeline()
        print("\nAll demonstrations completed successfully!")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
