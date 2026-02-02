import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_loss, competition_metric
from library.data import get_loaders
from library.model import RNAModel
from library.train import train_epoch, validate, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demo Pipeline...")

    # =========================================================================
    # 1. Configuration Overrides for Speed and Demonstration
    # =========================================================================
    # We modify the Config class attributes directly to affect the global state
    # used by the library modules.
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Set paths to a specific demo directory in working
    demo_dir = os.path.join(Config.INPUT_DIR, "../working/demo_execution")
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Pipeline Verification
    # =========================================================================
    print("\n--- Verifying Data Pipeline ---")
    # This will trigger data processing from metadata files since cache dir is new
    train_loader, val_loader, test_loader = get_loaders(debug=Config.DEBUG)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    bpp_indices = batch["bpp_indices"].to(device)
    targets = batch["targets"].to(device)

    # Assertions
    # Inputs: (Batch, Seq_Len, Channels=14)
    assert inputs.dim() == 3
    assert inputs.shape[1] == Config.SEQ_LEN
    assert inputs.shape[2] == Config.INPUT_CHANNELS

    # BPP Indices: (Batch, Seq_Len)
    assert bpp_indices.dim() == 2
    assert bpp_indices.shape[1] == Config.SEQ_LEN

    # Targets: (Batch, Seq_Len, Targets=5)
    assert targets.dim() == 3
    assert targets.shape[1] == Config.SEQ_LEN
    assert targets.shape[2] == Config.OUTPUT_CHANNELS

    print("Data shapes verified successfully.")

    # =========================================================================
    # 3. Model Initialization & Forward Pass
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")
    model = RNAModel(config=Config).to(device)

    # Run forward pass
    preds = model(inputs, bpp_indices)

    # Assert output shape matches targets
    assert (
        preds.shape == targets.shape
    ), f"Model output shape {preds.shape} mismatch with targets {targets.shape}"

    print("Model forward pass successful.")

    # =========================================================================
    # 4. Metric Verification
    # =========================================================================
    print("\n--- Verifying Metrics ---")
    # Calculate loss
    loss_val = mcrmse_loss(preds, targets)
    score_val = competition_metric(preds, targets)

    # Assert scalar return
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.ndim == 0
    assert isinstance(score_val, float)
    assert score_val >= 0.0

    print(f"Initial Loss: {loss_val.item():.4f}")
    print(f"Initial Score: {score_val:.4f}")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n--- Running Training Loop (2 Epochs) ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Score={val_score:.4f}")

        # Basic assertion that training returns valid numbers
        assert not np.isnan(train_loss)
        assert not np.isnan(val_score)

    # Save the model (simulating the best model save)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    print("Model saved.")

    # =========================================================================
    # 6. Inference & Submission Generation
    # =========================================================================
    print("\n--- Generating Submission ---")
    # Load model state to verify loading works
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Generate submission
    generate_submission(model, test_loader, device)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Expected rows: Num_Test_Samples (50 in debug) * Seq_Len (107)
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    # Verify id_seqpos format (e.g., id_xxxxx_0)
    sample_id_seqpos = df_sub.iloc[0]["id_seqpos"]
    assert (
        "_" in sample_id_seqpos and sample_id_seqpos.split("_")[-1].isdigit()
    ), f"Invalid id_seqpos format: {sample_id_seqpos}"

    print("Submission verification passed.")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
