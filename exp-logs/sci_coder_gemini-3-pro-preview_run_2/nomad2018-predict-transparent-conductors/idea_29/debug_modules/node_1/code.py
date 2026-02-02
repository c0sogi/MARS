import os
import shutil
import torch
import pandas as pd
import numpy as np
import sys

# Import the provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def run_demo():
    print("Starting Library Usage Demo...")

    # ==========================================
    # 1. Setup Demo Environment
    # ==========================================
    # Define temporary directories for this demo to avoid overwriting real work
    DEMO_WORK_DIR = "./working/demo_run"
    DEMO_META_DIR = "./working/demo_metadata"
    DEMO_CACHE_DIR = os.path.join(DEMO_WORK_DIR, "cache")
    DEMO_CHECKPOINT_DIR = os.path.join(DEMO_WORK_DIR, "checkpoints")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_WORK_DIR, "submission")

    os.makedirs(DEMO_META_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    print(f"Created demo directories at {DEMO_WORK_DIR}")

    # ==========================================
    # 2. Create Subset Metadata (Optimization)
    # ==========================================
    # We use a small subset of the real metadata to make data loading fast
    print("Creating subset metadata for speed...")

    # Read original metadata
    orig_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Take top 20 samples for train, 10 for val, 10 for test
    demo_train = orig_train.head(20)
    demo_val = orig_val.head(10)
    demo_test = orig_test.head(10)

    # Save to demo metadata dir
    demo_train_path = os.path.join(DEMO_META_DIR, "train_metadata.csv")
    demo_val_path = os.path.join(DEMO_META_DIR, "val_metadata.csv")
    demo_test_path = os.path.join(DEMO_META_DIR, "test_metadata.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # ==========================================
    # 3. Override Config (Optimization)
    # ==========================================
    print("Overriding config parameters...")
    # Override paths
    config.WORKING_DIR = DEMO_WORK_DIR
    config.CACHE_DIR = DEMO_CACHE_DIR
    config.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR
    config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR

    config.TRAIN_METADATA_PATH = demo_train_path
    config.VAL_METADATA_PATH = demo_val_path
    config.TEST_METADATA_PATH = demo_test_path

    # Override training params for speed
    config.MAX_EPOCHS = 2
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    config.HIDDEN_DIM = 32  # Reduce model size for speed
    config.NUM_LAYERS = 2  # Reduce depth

    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    # ==========================================
    # 4. Test Utility Functions
    # ==========================================
    print("\nTesting Utility Functions...")

    # Test RMSLE
    y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred = torch.tensor([[1.1, 1.9], [3.2, 3.8]])
    rmsle = utils.compute_rmsle(y_pred, y_true)
    print(f"  Computed RMSLE: {rmsle:.4f}")
    assert isinstance(rmsle, float), "RMSLE should return a float"
    assert rmsle >= 0, "RMSLE must be non-negative"

    # Test StandardScaler
    scaler = utils.StandardScaler()
    data_tensor = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler.fit(data_tensor)
    transformed = scaler.transform(data_tensor)

    # Check standardization (mean ~ 0, std ~ 1)
    assert torch.allclose(
        transformed.mean(dim=0), torch.zeros(2), atol=1e-6
    ), "Scaler mean not zero"
    # std of [1,2,3] is 1.0. (1-2)/1 = -1, (2-2)/1 = 0, (3-2)/1 = 1.
    expected_transformed = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(
        transformed, expected_transformed, atol=1e-6
    ), "Scaler transform incorrect"

    # Check inverse
    inversed = scaler.inverse_transform(transformed)
    assert torch.allclose(
        inversed, data_tensor, atol=1e-6
    ), "Scaler inverse transform failed"
    print("  StandardScaler logic verified.")

    # ==========================================
    # 5. Test Data Loading
    # ==========================================
    print("\nTesting Data Loading...")
    # This will process the subset geometry files and cache them
    train_loader, val_loader, test_loader, target_scaler = data.get_dataloaders(
        load_cached_data=False
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # Check a single batch
    batch = next(iter(train_loader))
    print(f"  Batch structure: {batch}")
    assert hasattr(batch, "x"), "Batch missing node features"
    assert hasattr(batch, "edge_index"), "Batch missing edge index"
    assert hasattr(batch, "edge_attr"), "Batch missing edge attributes"
    assert hasattr(batch, "y"), "Batch missing targets"
    assert (
        batch.y.shape[1] == 2
    ), f"Target shape mismatch. Expected 2, got {batch.y.shape[1]}"
    print("  Data loading verified.")

    # ==========================================
    # 6. Test Model Architecture
    # ==========================================
    print("\nTesting Model Architecture...")
    device = config.DEVICE
    # Initialize model with reduced dimensions
    net = model.LP_RA_CGN(
        node_input_dim=100, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS
    ).to(device)

    batch = batch.to(device)

    # Forward pass
    output = net(batch)
    print(f"  Model output shape: {output.shape}")

    assert output.shape == (batch.num_graphs, 2), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"
    print("  Model forward pass verified.")

    # ==========================================
    # 7. Test Training Loop
    # ==========================================
    print("\nTesting Training Loop...")

    trainer = train.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        target_scaler=target_scaler,
        device=device,
    )

    # Run a short training cycle
    trainer.fit(epochs=config.MAX_EPOCHS)

    # Check if checkpoint was saved
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file not created"
    print("  Training loop execution verified.")

    # ==========================================
    # 8. Test Inference / Submission
    # ==========================================
    print("\nTesting Inference and Submission...")

    # Load best model
    net.load_state_dict(torch.load(checkpoint_path))

    # Generate submission
    model.generate_submission(net, test_loader, target_scaler, device)

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not created"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"  Submission rows: {len(sub_df)}")
    assert len(sub_df) == len(demo_test), "Submission row count mismatch"
    assert "id" in sub_df.columns, "Submission missing 'id' column"
    assert (
        "formation_energy_ev_natom" in sub_df.columns
    ), "Submission missing formation energy column"
    assert (
        "bandgap_energy_ev" in sub_df.columns
    ), "Submission missing bandgap energy column"

    # Check for non-negative values (physical constraint applied in generate_submission)
    assert (
        sub_df["formation_energy_ev_natom"] >= 0
    ).all(), "Negative formation energy predictions found"
    assert (
        sub_df["bandgap_energy_ev"] >= 0
    ).all(), "Negative bandgap energy predictions found"

    print("  Inference verified.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nDemo FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
