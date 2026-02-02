import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import the provided library modules
import library.config as config
import library.utils as utils
import library.features as features
import library.data as data
import library.model as model
import library.train as train


def create_mini_datasets(n_breaths_train=20, n_breaths_val=10, n_breaths_test=10):
    """
    Creates small subsets of the original data to allow for rapid demonstration
    and testing of the pipeline.
    """
    print("--- Creating Mini Datasets for Speed Optimization ---")
    seq_len = config.SEQ_LEN

    # Define paths for mini datasets
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")

    # Read chunks from metadata files
    # We read (n_breaths * seq_len) rows
    train_chunk = pd.read_csv(config.TRAIN_CSV, nrows=n_breaths_train * seq_len)
    val_chunk = pd.read_csv(config.VAL_CSV, nrows=n_breaths_val * seq_len)
    test_chunk = pd.read_csv(config.TEST_CSV, nrows=n_breaths_test * seq_len)

    # Save to working directory
    train_chunk.to_csv(mini_train_path, index=False)
    val_chunk.to_csv(mini_val_path, index=False)
    test_chunk.to_csv(mini_test_path, index=False)

    print(f"Created mini_train.csv: {train_chunk.shape}")
    print(f"Created mini_val.csv:   {val_chunk.shape}")
    print(f"Created mini_test.csv:  {test_chunk.shape}")

    return mini_train_path, mini_val_path, mini_test_path


def patch_config(train_path, val_path, test_path):
    """
    Runtime patching of the config module to use mini datasets and faster settings.
    """
    print("--- Patching Configuration ---")
    # Patch paths
    config.TRAIN_CSV = train_path
    config.VAL_CSV = val_path
    config.TEST_CSV = test_path

    # Patch training settings for speed
    config.EPOCHS = 1
    config.BATCH_SIZE = 8  # Small batch size for mini dataset
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Use a separate cache dir for this demo to avoid messing with real training artifacts
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "mini_cache")
    config.MODEL_SAVE_PATH = os.path.join(config.WORKING_DIR, "mini_model.pth")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "mini_submission.csv")

    print(f"Config patched: EPOCHS={config.EPOCHS}, BATCH_SIZE={config.BATCH_SIZE}")


def demo_utils():
    print("\n=== Demonstrating Library: utils ===")

    # Test Seeding
    utils.seed_everything(42)
    r1 = np.random.rand()
    utils.seed_everything(42)
    r2 = np.random.rand()
    assert r1 == r2, "seed_everything did not ensure deterministic numpy behavior"
    print("Verification Passed: seed_everything ensures reproducibility.")

    # Test Device
    device = utils.get_device()
    print(f"Device detected: {device}")
    assert isinstance(device, torch.device)


def demo_features():
    print("\n=== Demonstrating Library: features ===")

    # Create a dummy dataframe representing 1 breath (80 steps)
    # R=5, C=10, u_in=10 constant
    seq_len = config.SEQ_LEN
    dummy_data = {
        config.ID_COL: np.arange(seq_len),
        config.BREATH_ID_COL: np.zeros(seq_len, dtype=int),
        config.TIME_COL: np.linspace(0, 2.5, seq_len),
        "R": np.full(seq_len, 5),
        "C": np.full(seq_len, 10),
        "u_in": np.full(seq_len, 10.0),
        "u_out": np.zeros(seq_len, dtype=int),
        "pressure": np.zeros(seq_len),
    }
    df = pd.DataFrame(dummy_data)

    # Test add_physics_features
    df_phys = features.add_physics_features(df.copy())

    # Verify 'dt'
    expected_dt = df[config.TIME_COL].diff().fillna(0).values
    np.testing.assert_allclose(
        df_phys["dt"].values, expected_dt, err_msg="dt calculation incorrect"
    )

    # Verify 'flow_interaction' = u_in * R
    expected_flow_int = df["u_in"] * df["R"]
    np.testing.assert_allclose(
        df_phys["flow_interaction"].values,
        expected_flow_int,
        err_msg="flow_interaction incorrect",
    )

    # Test encode_categoricals
    df_cat = features.encode_categoricals(df.copy())
    # R=5 should map to 0 based on features.R_MAP
    assert df_cat["R_idx"].iloc[0] == 0, "Categorical encoding for R failed"
    # C=10 should map to 0 based on features.C_MAP
    assert df_cat["C_idx"].iloc[0] == 0, "Categorical encoding for C failed"

    print("Verification Passed: Feature engineering logic is correct.")


def demo_data_pipeline():
    print("\n=== Demonstrating Library: data ===")

    # Load data loaders (this triggers features.prepare_datasets)
    # We force reload to ensure it uses our patched config paths
    if os.path.exists(config.CACHE_DIR):
        shutil.rmtree(config.CACHE_DIR)

    train_loader, val_loader, test_loader = data.get_data_loaders(
        load_cached_data=False,
        batch_size=config.BATCH_SIZE,
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches:   {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Verify keys
    required_keys = ["x_cont", "x_cat", "x_phys", "u_out", "ids", "y"]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Verify shapes
    # x_cont: (Batch, Seq, Feats)
    assert batch["x_cont"].shape[0] == config.BATCH_SIZE
    assert batch["x_cont"].shape[1] == config.SEQ_LEN
    assert batch["x_cont"].shape[2] == len(config.CONTINUOUS_FEATURES)

    # x_cat: (Batch, Seq, 2)
    assert batch["x_cat"].shape[2] == 2

    print("Verification Passed: DataLoaders yield correct batch shapes.")
    return train_loader, val_loader, test_loader


def demo_model_and_loss():
    print("\n=== Demonstrating Library: model & loss ===")

    device = utils.get_device()
    net = model.PhysicsResidualModel().to(device)

    # Create dummy inputs
    bs = 4
    seq = config.SEQ_LEN
    n_cont = len(config.CONTINUOUS_FEATURES)
    n_phys = len(config.PHYSICS_FEATURES)

    x_cont = torch.randn(bs, seq, n_cont).to(device)
    x_cat = torch.zeros(bs, seq, 2, dtype=torch.long).to(device)  # Indices 0
    x_phys = torch.randn(bs, seq, n_phys).to(device)

    # Forward pass
    output = net(x_cont, x_cat, x_phys)

    # Check output shape: (Batch, Seq)
    assert output.shape == (
        bs,
        seq,
    ), f"Model output shape mismatch. Expected {(bs, seq)}, got {output.shape}"
    print(f"Model Forward Pass successful. Output shape: {output.shape}")

    # Test Loss
    criterion = train.MaskedL1Loss()

    # Case 1: Perfect prediction -> Loss should be 0
    u_out = torch.zeros(bs, seq).to(device)  # All inspiratory
    loss_zero = criterion(output, output, u_out)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0).to(device)
    ), "Loss should be 0 for perfect prediction"

    # Case 2: Error only in expiratory phase (u_out=1) -> Loss should be 0 (masked)
    u_out_exp = torch.ones(bs, seq).to(device)  # All expiratory
    # Create large error
    target_diff = output + 100.0
    loss_masked = criterion(output, target_diff, u_out_exp)
    # The loss function adds 1e-8 to denominator, so result is 0/1e-8 = 0
    assert torch.isclose(
        loss_masked, torch.tensor(0.0).to(device)
    ), "Loss should be 0 when all errors are masked"

    print("Verification Passed: MaskedL1Loss logic is correct.")

    return net


def demo_training_loop(net, train_loader, val_loader):
    print("\n=== Demonstrating Library: train (Epoch Execution) ===")

    device = utils.get_device()
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
    )
    criterion = train.MaskedL1Loss()

    # Run 1 Train Epoch
    print("Running training epoch...")
    train_loss = train.train_epoch(
        net, train_loader, optimizer, scheduler, device, criterion
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert train_loss >= 0, "Training loss should be non-negative"

    # Run 1 Validation Epoch
    print("Running validation epoch...")
    val_mae = train.validate_epoch(net, val_loader, device, criterion)
    print(f"Val MAE: {val_mae:.4f}")
    assert val_mae >= 0, "Validation MAE should be non-negative"

    # Save model for prediction step
    torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")


def demo_prediction(net):
    print("\n=== Demonstrating Library: train (Prediction) ===")

    # We can use the predict_and_submit function from library.train
    # It loads the model from disk and runs inference on test set

    # Ensure test loader is available via cache (it was prepared in demo_data_pipeline)
    # The function re-initializes loaders, which is fine.

    train.predict_and_submit()

    # Verify submission file
    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission generated at {config.SUBMISSION_PATH}")
        print(f"Submission Shape: {sub_df.shape}")

        # Verify columns
        assert list(sub_df.columns) == ["id", "pressure"], "Submission columns mismatch"
        assert not sub_df.isnull().any().any(), "Submission contains NaNs"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # 1. Setup Environment
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Create Mini Datasets & Patch Config
    mini_train, mini_val, mini_test = create_mini_datasets()
    patch_config(mini_train, mini_val, mini_test)

    # 3. Run Demonstrations
    demo_utils()
    demo_features()

    # Data Pipeline
    tr_loader, val_loader, te_loader = demo_data_pipeline()

    # Model & Loss
    model_instance = demo_model_and_loss()

    # Training Loop
    demo_training_loop(model_instance, tr_loader, val_loader)

    # Prediction
    demo_prediction(model_instance)

    print("\n=== All Demonstrations Completed Successfully ===")
