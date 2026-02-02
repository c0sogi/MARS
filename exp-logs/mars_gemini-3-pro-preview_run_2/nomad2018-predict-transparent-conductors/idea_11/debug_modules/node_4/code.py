import os
import sys
import shutil
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Cite debug_lesson_8: Reload Modules When Patching Libraries in Persistent Environments
for module in list(sys.modules.keys()):
    if module.startswith("library"):
        del sys.modules[module]

# Import library components
# Note: We modify Config attributes before importing other modules where possible,
# or modify them on the singleton instance since Python modules are cached.
from library.config import Config
from library.utils import set_seed, compute_rmsle, StandardScaler
from library.data import get_dataset, CrystalGraphDataset
from library.model import MSR_CGCNN
from library.train import train_model, Trainer


def run_demo():
    print("=== Starting MSR-CGCNN Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Override Config defaults to ensure the demo runs quickly and uses the correct paths
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Model saving paths
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.TARGET_SCALER_PATH = os.path.join(Config.CACHE_DIR, "target_scaler.pth")
    Config.GLOBAL_SCALER_PATH = os.path.join(Config.CACHE_DIR, "global_scaler.pth")

    # Training hyperparams for speed
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples for speed

    # Re-run setup to create new directories
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # -------------------------------------------------------------------------
    # 2. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test RMSLE
    y_true = np.array([1.0, 10.0, 100.0])
    y_pred = np.array([1.1, 9.5, 105.0])
    rmsle = compute_rmsle(y_true, y_pred)
    print(f"Computed RMSLE: {rmsle:.4f}")
    assert rmsle >= 0, "RMSLE should be non-negative"

    # Test StandardScaler
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    scaler.fit(data)
    transformed = scaler.transform(data)
    inverse = scaler.inverse_transform(transformed)

    assert np.allclose(
        data, inverse
    ), "Scaler inverse transform failed to recover original data"
    print("StandardScaler logic verified.")

    # -------------------------------------------------------------------------
    # 3. Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n[3] Loading and Processing Data...")

    # Load training dataset (this will trigger processing and scaling)
    # debug=True forces it to use the subset defined in Config.DEBUG_SAMPLE_SIZE
    train_dataset = get_dataset("train", load_cached_data=False, debug=True)
    val_dataset = get_dataset("val", load_cached_data=False, debug=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Verify Data object structure
    sample_data = train_dataset[0]
    print("Sample Data Object:", sample_data)

    assert sample_data.x.dim() == 1, "Node features should be 1D (atomic numbers)"
    assert sample_data.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert sample_data.global_x.shape[1] == len(
        Config.GLOBAL_FEATURES
    ), "Global features dim mismatch"
    assert sample_data.y.shape[1] == len(Config.TARGET_COLS), "Target dim mismatch"

    print("Data processing verified.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSR_CGCNN(config=Config).to(device)

    # Create a dataloader for a single batch
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch = next(iter(train_loader)).to(device)

    # Run forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print(f"Batch output shape: {output.shape}")
    assert output.shape == (
        batch.num_graphs,
        len(Config.TARGET_COLS),
    ), f"Expected output shape {(batch.num_graphs, len(Config.TARGET_COLS))}, got {output.shape}"

    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop...")

    # We use the high-level train_model function which handles optimizer, loop, and saving
    trained_model = train_model(
        max_epochs=Config.MAX_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.PATIENCE,
        debug=True,  # Ensure we use the debug setting
    )

    print("Training loop execution successful.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Load test dataset
    # Note: Test dataset targets are placeholders (zeros)
    test_dataset = get_dataset("test", load_cached_data=False, debug=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load best model
    best_model = MSR_CGCNN(config=Config).to(device)
    checkpoint = torch.load(
        Config.BEST_MODEL_PATH, map_location=device, weights_only=False
    )
    best_model.load_state_dict(checkpoint)
    best_model.eval()

    # Load scalers for inverse transformation
    target_scaler = StandardScaler()
    target_scaler.load_state_dict(
        torch.load(Config.TARGET_SCALER_PATH, weights_only=False)
    )

    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = best_model(batch)
            predictions.append(out.cpu().numpy())
            ids.extend(batch.material_id.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Inverse transform predictions to original scale
    real_predictions = target_scaler.inverse_transform(predictions)

    # Create submission dataframe
    import pandas as pd

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": real_predictions[:, 0],
            "bandgap_energy_ev": real_predictions[:, 1],
        }
    )

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 predictions:")
    print(submission_df.head())

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Raise to ensure non-zero exit code on failure
        raise e
