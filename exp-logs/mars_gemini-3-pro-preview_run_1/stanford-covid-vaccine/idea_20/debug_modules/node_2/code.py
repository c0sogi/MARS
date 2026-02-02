import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    generate_submission_file,
    load_checkpoint,
    mcrmse_metric,
)
from library.data import (
    process_data,
    RNADataset,
    get_dataloaders,
    parse_structure,
    tokenize_sequence,
)
from library.model import Net, SinusoidalPositionalEmbedding
from library.loss import MaskedMSELoss, calculate_mcrmse
from library.train import Trainer
from library.predict import Predictor


def run_demo():
    # ==========================================
    # 1. Configuration and Setup
    # ==========================================
    print(">>> 1. Configuring environment for demo...")

    # Override Config for speed
    Config.debug = True
    Config.debug_subset_size = 50  # Use only 50 samples
    Config.epochs = 1
    Config.batch_size = 4
    Config.working_dir = "./working/demo_run"
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_path = os.path.join(
        Config.working_dir, "submission", "submission.csv"
    )

    # Clean up previous demo run if exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(os.path.dirname(Config.model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    seed_everything(Config.seed)
    logger = get_logger("Demo", log_file=os.path.join(Config.working_dir, "demo.log"))
    logger.info("Demo configuration applied.")

    # ==========================================
    # 2. Data Processing Logic Verification
    # ==========================================
    print(">>> 2. Verifying Data Processing Logic...")

    # Test Structure Parsing
    # Structure: (()) -> Indices: 0 paired with 3, 1 paired with 2
    # Distances: idx 0 -> (3-0)=3, idx 3 -> (0-3)=-3
    dummy_struct = "(())"
    dists = parse_structure(dummy_struct, seq_len=4)
    expected_dists = np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float32)
    np.testing.assert_array_equal(
        dists, expected_dists, err_msg="Structure parsing logic failed."
    )
    print("   [Pass] parse_structure logic verified.")

    # Test Data Loading Pipeline
    print("   Processing data (this calls library.data.process_data)...")
    # This will create cache files in working/demo_run
    train_seq, train_loop, train_dist, train_tgt, train_mask = process_data(
        "train", load_cached_data=False
    )

    assert len(train_seq) == Config.debug_subset_size
    assert train_seq.shape == (Config.debug_subset_size, Config.seq_len)
    assert train_tgt.shape == (Config.debug_subset_size, Config.seq_len, 3)
    assert train_mask.shape == (Config.debug_subset_size, Config.seq_len)
    print(f"   [Pass] Data shapes verified: {train_seq.shape}")

    # Create Loaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Fetch one batch
    batch = next(iter(train_loader))
    print(f"   [Pass] Batch keys: {batch.keys()}")
    assert "sequence" in batch and "target" in batch
    assert batch["sequence"].shape == (Config.batch_size, Config.seq_len)

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print(">>> 3. Verifying Model Architecture...")

    model = Net().to(Config.device)

    # Test Embedding
    # Sinusoidal embedding should return (Batch, Seq, distance_dim)
    dist_layer = SinusoidalPositionalEmbedding(dim=Config.distance_dim).to(
        Config.device
    )
    dummy_dist = torch.zeros(2, 10).to(Config.device)
    emb_out = dist_layer(dummy_dist)
    assert emb_out.shape == (2, 10, Config.distance_dim)
    print("   [Pass] SinusoidalPositionalEmbedding shape verified.")

    # Test Full Forward Pass
    seq = batch["sequence"].to(Config.device)
    loop = batch["loop_type"].to(Config.device)
    dist = batch["distance"].to(Config.device)

    output = model(seq, loop, dist)

    # Output should be (Batch, Seq_Len, n_targets=3)
    assert output.shape == (Config.batch_size, Config.seq_len, 3)
    assert not torch.isnan(output).any(), "Model output contains NaNs"
    print(f"   [Pass] Model forward pass successful. Output shape: {output.shape}")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print(">>> 4. Verifying Loss Function Logic...")

    criterion = MaskedMSELoss()

    # Case 1: Perfect prediction -> Loss 0
    dummy_pred = torch.ones(2, 5, 3)
    dummy_target = torch.ones(2, 5, 3)
    dummy_mask = torch.ones(2, 5, dtype=torch.bool)
    loss = criterion(dummy_pred, dummy_target, dummy_mask)
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), "Loss should be 0 for perfect prediction"

    # Case 2: Known error
    # Pred = 1, Target = 0. Error = 1^2 = 1.
    dummy_target_zero = torch.zeros(2, 5, 3)
    loss = criterion(dummy_pred, dummy_target_zero, dummy_mask)
    assert torch.isclose(loss, torch.tensor(1.0)), "Loss should be 1.0"

    # Case 3: Masking
    # If mask is False, those values shouldn't count
    dummy_mask_mixed = torch.zeros(2, 5, dtype=torch.bool)
    dummy_mask_mixed[:, 0] = True  # Only first column valid
    # Pred=1, Target=0 at valid pos -> error 1.
    # Pred=100, Target=0 at invalid pos -> error ignored.
    dummy_pred_bad = dummy_pred.clone()
    dummy_pred_bad[:, 1:] = 100.0

    loss = criterion(dummy_pred_bad, dummy_target_zero, dummy_mask_mixed)
    assert torch.isclose(
        loss, torch.tensor(1.0)
    ), "Masking logic failed, loss affected by masked values"

    # Metric Check
    metric_val = calculate_mcrmse(dummy_pred, dummy_target_zero, dummy_mask)
    # RMSE of 1 is 1.
    assert torch.isclose(metric_val, torch.tensor(1.0))
    print("   [Pass] MaskedMSELoss and MCRMSE logic verified.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print(">>> 5. Running Training Loop Demo...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=Config.device,
        logger=logger,
    )

    # Run fit (1 epoch as configured)
    trainer.fit(epochs=Config.epochs, patience=1)

    assert os.path.exists(
        Config.model_save_path
    ), "Best model checkpoint was not saved."
    print("   [Pass] Training loop completed and model saved.")

    # ==========================================
    # 6. Prediction and Submission Demonstration
    # ==========================================
    print(">>> 6. Running Prediction and Submission Demo...")

    predictor = Predictor(
        model_path=Config.model_save_path, device=Config.device, logger=logger
    )

    # Run inference
    predictions = predictor.predict(test_loader)

    # Check predictions shape: (N_test_samples, Seq_Len, 3)
    # Note: test loader might drop last if configured, but here it shouldn't.
    # In debug mode, test set is also subsetted.
    n_test_samples = len(
        pd.read_parquet(Config.test_file).head(Config.debug_subset_size)
    )
    assert predictions.shape == (n_test_samples, Config.seq_len, 3)

    # Generate Submission
    df_test = pd.read_parquet(Config.test_file).head(Config.debug_subset_size)
    ids = df_test["id"].tolist()
    sequences = df_test["sequence"].tolist()

    generate_submission_file(ids, sequences, predictions, Config.submission_path)

    assert os.path.exists(Config.submission_path), "Submission file not created."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.submission_path)
    expected_rows = n_test_samples * Config.seq_len
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"
    assert "reactivity" in sub_df.columns
    assert "deg_Mg_pH10" in sub_df.columns

    print(f"   [Pass] Submission generated with {len(sub_df)} rows.")
    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
