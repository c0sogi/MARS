import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to path
sys.path.append(".")

# Import library modules
from library.config import Config
import library.dataset
from library.dataset import preprocess_data, get_dataloaders
from library.network import HybridModel
from library.engine import run_training, generate_submission, set_seed

# ------------------------------------------------------------------------------
# Monkey-patching to fix dataset.py logic
# ------------------------------------------------------------------------------
# The provided dataset.py attempts to load f_00 through f_30 as continuous features.
# However, f_27 is a string sequence feature. Including it in continuous_data
# causes a ValueError during float conversion and a dimension mismatch in the model (31 vs 30).
# We patch pd.read_csv in library.dataset to rename f_27, ensuring it is
# excluded from the auto-detected continuous columns (which look for "f_XX").
# ------------------------------------------------------------------------------

_original_read_csv = library.dataset.pd.read_csv


def _patched_read_csv(filepath_or_buffer, *args, **kwargs):
    df = _original_read_csv(filepath_or_buffer, *args, **kwargs)
    # Rename f_27 to prevent it from being picked up by f_{i:02d} loop in dataset.py
    if "f_27" in df.columns:
        df.rename(columns={"f_27": "seq_27"}, inplace=True)
    return df


# Apply the patch to the library's pandas instance
library.dataset.pd.read_csv = _patched_read_csv


def main():
    print("Starting execution...")
    warnings.filterwarnings("ignore")

    # 1. Configuration
    set_seed(42)

    # Adjust Config for the patch (Sequence feature is now named seq_27)
    Config.SEQUENCE_FEATURE = "seq_27"

    # Adjust Config for speed (Demo requirements)
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2048
    Config.EMBED_DIM = 16
    Config.TRANSFORMER_LAYERS = 1
    Config.TRANSFORMER_HEADS = 2
    Config.BACKBONE_WIDTHS = [64, 32]

    # 2. Data Processing & Verification
    print("\n[Step 1] Processing Data...")

    # Force reload to ensure our patch takes effect and we don't use old cache
    data = preprocess_data(load_cached_data=False)

    # Verify dimensions
    # Continuous: Should be 30 features (f_00..f_30 excluding f_27)
    # If the patch failed, this would be 31 or the code would have crashed earlier.
    assert (
        data["continuous_data"].shape[1] == 30
    ), f"Expected 30 continuous features, got {data['continuous_data'].shape[1]}"

    # Sequence: Should be 10 chars
    assert (
        data["sequence_data"].shape[1] == 10
    ), f"Expected 10 sequence features, got {data['sequence_data'].shape[1]}"

    print("Data processed and verified.")

    # 3. Model Verification
    print("\n[Step 2] Verifying Model...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel().to(device)

    # Test forward pass with one batch
    batch = next(iter(train_loader))
    cont = batch["continuous"].to(device)
    seq = batch["sequence"].to(device)

    with torch.no_grad():
        out = model(cont, seq)

    assert out.shape == (cont.shape[0], 1), "Model output shape mismatch"
    print("Model initialized and forward pass successful.")

    # 4. Training
    print("\n[Step 3] Running Training Loop...")
    # run_training handles the loop and saving best_model.pth
    trained_model = run_training()

    assert os.path.exists(Config.MODEL_PATH), "Model file not generated."
    print("Training finished.")

    # 5. Inference
    print("\n[Step 4] Generating Submission...")
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated."

    # Validate submission format
    sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub.shape == (100000, 2), f"Submission shape mismatch: {sub.shape}"
    assert list(sub.columns) == ["id", "target"], "Submission columns mismatch"

    # Validate probabilities
    preds = sub["target"].values
    assert (preds >= 0).all() and (preds <= 1).all(), "Predictions out of range [0, 1]"

    print("\nExecution completed successfully.")


if __name__ == "__main__":
    main()
