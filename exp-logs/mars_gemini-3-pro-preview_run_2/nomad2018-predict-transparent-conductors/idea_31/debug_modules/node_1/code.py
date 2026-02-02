import os
import shutil
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.data import get_dataloaders, CrystalGraphDataset
from library.model import SRACGN
from library.utils import set_seed, StandardScaler, save_checkpoint, load_checkpoint
from library.train import train_one_epoch, evaluate, fit_and_cache_scaler


def run_demo():
    print("=== Starting SRA-CGN Library Demo ===")

    # 1. Setup Configuration for Demo
    # We modify the Config class attributes directly to affect all modules
    print("\n[1] Configuring environment...")

    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Create subdirectories for metadata
    DEMO_METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)

    # Update Config paths
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Update Hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.ATOM_EMBEDDING_DIM = 32  # Reduced from 128
    Config.NUM_INTERACTION_BLOCKS = 2  # Reduced from 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.TRAIN_METADATA = os.path.join(DEMO_METADATA_DIR, "train_metadata.csv")
    Config.VAL_METADATA = os.path.join(DEMO_METADATA_DIR, "val_metadata.csv")
    Config.TEST_METADATA = os.path.join(DEMO_METADATA_DIR, "test_metadata.csv")

    Config.setup()
    set_seed(Config.SEED)

    # 2. Prepare Data Subset
    print("\n[2] Creating data subsets for speed...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Take a small sample (e.g., 10 samples each)
    demo_train = orig_train.head(10)
    demo_val = orig_val.head(5)
    demo_test = orig_test.head(5)

    # Save to demo metadata directory
    demo_train.to_csv(Config.TRAIN_METADATA, index=False)
    demo_val.to_csv(Config.VAL_METADATA, index=False)
    demo_test.to_csv(Config.TEST_METADATA, index=False)

    print(
        f"Created subset metadata: Train={len(demo_train)}, Val={len(demo_val)}, Test={len(demo_test)}"
    )

    # 3. Data Loading
    print("\n[3] Testing Data Loading and Processing...")
    # Force reprocessing by setting load_cached=False initially or ensuring cache doesn't exist
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached=False,  # Force processing from XYZ files
    )

    # Verify a batch
    sample_batch = next(iter(train_loader))
    print(f"Sample Batch: {sample_batch}")
    assert sample_batch.x.ndim == 1, "Node features should be 1D (atomic numbers)"
    assert (
        sample_batch.edge_index.shape[0] == 2
    ), "Edge index should have shape (2, num_edges)"
    assert sample_batch.y.shape == (sample_batch.num_graphs, 2), "Target shape mismatch"
    print("Data loading successful.")

    # 4. Model Initialization
    print("\n[4] Initializing SRACGN Model...")
    device = Config.DEVICE
    model = SRACGN(config=Config).to(device)
    print(model)

    # Forward pass check
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        output = model(sample_batch)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        sample_batch.num_graphs,
        2,
    ), f"Expected output shape {(sample_batch.num_graphs, 2)}, got {output.shape}"
    print("Forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop...")

    # Fit Scaler
    scaler = fit_and_cache_scaler(train_loader, Config.CACHE_DIR, load_cached=False)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"--- Epoch {epoch} ---")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, scaler, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: {val_metrics}")

        # Basic assertion to ensure loss is not NaN
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Checkpointing
    print("\n[6] Testing Checkpointing...")
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "demo_model.pth")
    save_checkpoint(model, optimizer, Config.NUM_EPOCHS, val_loss, scaler, ckpt_path)

    assert os.path.exists(ckpt_path), "Checkpoint file was not created"

    # Load checkpoint
    loaded_model = SRACGN(config=Config).to(device)
    loaded_scaler = StandardScaler()
    checkpoint = load_checkpoint(loaded_model, None, ckpt_path, loaded_scaler)

    assert checkpoint is not None, "Failed to load checkpoint"
    assert "model_state_dict" in checkpoint
    assert np.array_equal(
        scaler.mean, loaded_scaler.mean
    ), "Scaler mean not restored correctly"
    print("Checkpointing successful.")

    # 7. Inference / Submission
    print("\n[7] Generating Submission...")
    loaded_model.eval()

    ids = []
    formation_preds = []
    bandgap_preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds_norm = loaded_model(batch)
            preds_raw = loaded_scaler.inverse_transform(preds_norm)
            preds_raw = torch.clamp(preds_raw, min=0.0)

            ids.extend(batch.material_id.cpu().numpy().flatten())
            formation_preds.extend(preds_raw[:, 0].cpu().numpy())
            bandgap_preds.extend(preds_raw[:, 1].cpu().numpy())

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": formation_preds,
            "bandgap_energy_ev": bandgap_preds,
        }
    )

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission generated with {len(submission_df)} rows.")
    print(submission_df.head())

    assert len(submission_df) == len(demo_test), "Submission length mismatch"
    assert os.path.exists(submission_path), "Submission file not found"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
