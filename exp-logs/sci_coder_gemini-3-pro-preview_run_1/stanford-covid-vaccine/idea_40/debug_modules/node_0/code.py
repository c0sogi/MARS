import os
import shutil
import warnings
import torch
import numpy as np
import pandas as pd

# Import provided library components
from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import InputNormedWideResBiGRU
from library.loss import MaskedMSELoss
from library.train import train_one_epoch, validate, generate_submission

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print(">>> 1. Configuring environment for demonstration...")

    # Override Config to use a temporary working directory and debug settings
    # This ensures the demo runs quickly and doesn't overwrite production files
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for verification

    # Create necessary directories
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print("    Configuration complete.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n>>> 2. Verifying Data Pipeline...")

    # Load dataloaders in debug mode (forces processing of small subset)
    # load_cached_data=False ensures we process from the provided metadata files
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))

    # Assert keys exist
    required_keys = ["sequence", "loop_type", "pair_dist", "target", "id"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Assert shapes
    # Sequence: [Batch, SeqLen]
    assert batch["sequence"].shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    # Target: [Batch, SeqLen, NumClasses]
    assert batch["target"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_CLASSES,
    )

    print(
        f"    Batch shapes verified: Sequence {batch['sequence'].shape}, Target {batch['target'].shape}"
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n>>> 3. Verifying Model Architecture...")

    model = InputNormedWideResBiGRU().to(device)

    # Move inputs to device
    seq = batch["sequence"].to(device)
    loop = batch["loop_type"].to(device)
    dist = batch["pair_dist"].to(device)
    targets = batch["target"].to(device)

    # Forward pass
    outputs = model(seq, loop, dist)

    # Verify output shape [Batch, SeqLen, NumClasses]
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_CLASSES)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Got {outputs.shape}, expected {expected_shape}"

    print("    Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n>>> 4. Verifying Loss Calculation...")

    criterion = MaskedMSELoss(scoring_length=Config.PRED_LEN)
    loss = criterion(outputs, targets)

    # Verify loss is a scalar and valid
    assert loss.dim() == 0, "Loss should be a scalar tensor"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"    Loss calculated successfully: {loss.item():.6f}")

    # -------------------------------------------------------------------------
    # 5. Training & Validation Loop
    # -------------------------------------------------------------------------
    print("\n>>> 5. Running Training/Validation Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.6f}")

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"    Validation MCRMSE: {val_score:.6f}")

    assert val_score >= 0, "Validation score should be non-negative"

    # -------------------------------------------------------------------------
    # 6. Metric Verification
    # -------------------------------------------------------------------------
    print("\n>>> 6. Verifying Metric (MCRMSE)...")

    # Create synthetic ground truth and predictions
    # Shape: (N, ScoredLen, Channels)
    dummy_true = np.zeros((10, 68, 3))
    dummy_pred = np.ones((10, 68, 3)) * 0.5  # Error is 0.5 everywhere

    # RMSE of 0.5 error is 0.5. MCRMSE is mean of RMSEs, so also 0.5
    calculated_metric = mcrmse(dummy_true, dummy_pred)

    assert np.isclose(
        calculated_metric, 0.5, atol=1e-5
    ), f"Metric calculation incorrect. Expected 0.5, got {calculated_metric}"

    print(f"    Metric verified correctly on synthetic data.")

    # -------------------------------------------------------------------------
    # 7. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n>>> 7. Generating Submission...")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify file content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: Debug subset size * Seq Length
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print(f"    Submission verified: {len(df_sub)} rows.")
    print("\n>>> Demonstration completed successfully.")
