import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.dataset import load_data, RNADataset
from library.model import (
    DenseAggregatedBiGRU,
    set_seed,
    generate_submission,
    masked_mse_loss,
    compute_mcrmse,
)
from library.trainer import Trainer


def run_demo():
    print("=" * 50)
    print("RNA Degradation Prediction: Library Usage Demo")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config to run in a temporary directory with minimal resources
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up any previous demo run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load training data (debug subset)
    # load_cached_data=False ensures we process from parquet for this demo
    train_dataset = load_data("train", debug=True, load_cached_data=False)

    # Assertions
    assert isinstance(train_dataset, RNADataset), "Returned object is not an RNADataset"
    assert (
        len(train_dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} samples, got {len(train_dataset)}"

    # Inspect a single sample
    sample = train_dataset[0]
    required_keys = ["sequence", "loop_type", "pair_dist", "targets", "mask", "id"]
    for key in required_keys:
        assert key in sample, f"Sample missing key: {key}"

    # Check tensor shapes
    seq_len = Config.SEQ_LEN
    assert sample["sequence"].shape == (
        seq_len,
    ), f"Sequence shape error: {sample['sequence'].shape}"
    assert sample["targets"].shape == (
        seq_len,
        3,
    ), f"Targets shape error: {sample['targets'].shape}"
    assert sample["mask"].shape == (
        seq_len,
    ), f"Mask shape error: {sample['mask'].shape}"

    print("    Data loaded successfully.")
    print(f"    Sample ID: {sample['id']}")
    print(f"    Sequence Shape: {sample['sequence'].shape}")

    # -------------------------------------------------------------------------
    # 3. Model Architecture & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    model = DenseAggregatedBiGRU(Config).to(device)

    # Create a dummy batch from the dataset
    batch_size = 2
    batch_indices = range(batch_size)

    sequences = torch.stack([train_dataset[i]["sequence"] for i in batch_indices]).to(
        device
    )
    loop_types = torch.stack([train_dataset[i]["loop_type"] for i in batch_indices]).to(
        device
    )
    pair_dists = torch.stack([train_dataset[i]["pair_dist"] for i in batch_indices]).to(
        device
    )

    # Run Forward Pass
    model.eval()
    with torch.no_grad():
        outputs = model(sequences, loop_types, pair_dists)

    # Check output shape: (Batch, Seq_Len, 3)
    expected_shape = (batch_size, seq_len, 3)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    print("    Forward pass successful.")
    print(f"    Output Shape: {outputs.shape}")

    # -------------------------------------------------------------------------
    # 4. Loss and Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Loss and Metric Calculations...")

    # Construct synthetic data
    # Mask: Valid for first 5 positions only
    dummy_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool).to(device)
    dummy_mask[:, :5] = True

    dummy_targets = torch.zeros((batch_size, seq_len, 3)).to(device)

    # Case A: Perfect Predictions
    dummy_preds_perfect = dummy_targets.clone()
    loss_zero = masked_mse_loss(dummy_preds_perfect, dummy_targets, dummy_mask)
    metric_zero = compute_mcrmse(dummy_preds_perfect, dummy_targets, dummy_mask)

    assert torch.isclose(
        loss_zero, torch.tensor(0.0).to(device)
    ), "Loss should be 0 for perfect predictions"
    assert torch.isclose(
        metric_zero, torch.tensor(0.0).to(device)
    ), "Metric should be 0 for perfect predictions"

    # Case B: Off-by-one Error
    # Add 1.0 error to valid positions
    dummy_preds_error = dummy_targets.clone()
    dummy_preds_error[:, :5, :] += 1.0

    loss_one = masked_mse_loss(dummy_preds_error, dummy_targets, dummy_mask)
    metric_one = compute_mcrmse(dummy_preds_error, dummy_targets, dummy_mask)

    # MSE of 1.0 is 1.0. RMSE of 1.0 is 1.0.
    assert torch.isclose(
        loss_one, torch.tensor(1.0).to(device)
    ), f"Expected Loss 1.0, got {loss_one.item()}"
    assert torch.isclose(
        metric_one, torch.tensor(1.0).to(device)
    ), f"Expected Metric 1.0, got {metric_one.item()}"

    print("    Loss and Metric logic verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Integration Test
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Integration Test)...")

    trainer = Trainer(Config)

    # Run fit (this uses load_data internally with debug=True based on our config/arg)
    best_score = trainer.fit(debug=True)

    # Verify artifacts
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not saved."
    assert best_score < float("inf"), "Trainer returned infinite score."

    print(f"    Training complete. Best Score: {best_score:.4f}")
    print(f"    Model saved to: {model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Load test data (debug subset)
    test_dataset = load_data("test", debug=True, load_cached_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Load best model state
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate submission
    generate_submission(model, test_loader, device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    # Verify Rows: Debug subset size (100) * Seq Len (107) = 10700
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, found {len(df_sub)}"

    # Verify Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    for col in expected_cols:
        assert col in df_sub.columns, f"Submission missing column: {col}"

    print("    Submission verification successful.")

    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
