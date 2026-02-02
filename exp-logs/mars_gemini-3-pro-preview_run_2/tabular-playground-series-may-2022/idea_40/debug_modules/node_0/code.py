import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import process_and_cache_data, get_dataloaders
from library.model import HybridNetwork
from library.train import run_training


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    print("Step 1: Setup and Seeding")
    seed_everything(Config.SEED)

    # We override the default epoch count and batch size for this demonstration
    # to ensure the script completes quickly while still exercising all code paths.
    DEMO_EPOCHS = 2
    DEMO_BATCH_SIZE = 2048  # Increased batch size for A100 efficiency

    # --------------------------------------------------------------------------
    # 2. Verify Data Processing Logic
    # --------------------------------------------------------------------------
    print("\nStep 2: Verifying Data Processing")
    # Force processing from scratch to verify the logic in library.dataset
    ids, X_cat, X_cont, y, id_to_idx = process_and_cache_data(load_cached_data=False)

    print(f"Processed Data Shapes:")
    print(f"  IDs: {ids.shape}")
    print(f"  Categorical (X_cat): {X_cat.shape}")
    print(f"  Continuous (X_cont): {X_cont.shape}")
    print(f"  Targets (y): {y.shape}")

    # Assertions to ensure data integrity
    assert (
        len(ids) == len(y) == len(X_cat) == len(X_cont)
    ), "Data array lengths mismatch"
    assert (
        X_cat.shape[1] == Config.SEQ_LEN
    ), f"Expected SEQ_LEN {Config.SEQ_LEN}, got {X_cat.shape[1]}"
    assert (
        X_cont.shape[1] == Config.NUM_CONT_FEATURES
    ), f"Expected NUM_CONT_FEATURES {Config.NUM_CONT_FEATURES}, got {X_cont.shape[1]}"

    # --------------------------------------------------------------------------
    # 3. Verify DataLoader Logic
    # --------------------------------------------------------------------------
    print("\nStep 3: Verifying DataLoaders")
    # This will reuse the cache created in Step 2
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=DEMO_BATCH_SIZE, load_cached_data=True
    )

    # Fetch one batch to verify tensor shapes
    cat_batch, cont_batch, target_batch = next(iter(train_loader))
    print(
        f"Batch Shapes -> Cat: {cat_batch.shape}, Cont: {cont_batch.shape}, Target: {target_batch.shape}"
    )

    assert cat_batch.shape == (DEMO_BATCH_SIZE, Config.SEQ_LEN)
    assert cont_batch.shape == (DEMO_BATCH_SIZE, Config.NUM_CONT_FEATURES)
    assert target_batch.shape == (DEMO_BATCH_SIZE,)

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\nStep 4: Verifying Model Architecture")
    device = Config.DEVICE
    model = HybridNetwork().to(device)

    # Create dummy input tensors on the appropriate device
    dummy_cat = torch.randint(0, Config.VOCAB_SIZE, (32, Config.SEQ_LEN)).to(device)
    dummy_cont = torch.randn(32, Config.NUM_CONT_FEATURES).to(device)

    # Perform a forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        output = model(dummy_cat, dummy_cont)

    print(f"Model Output Shape: {output.shape}")

    # Assertions for model output
    assert output.shape == (
        32,
    ), "Model output shape mismatch (should be 1D tensor of batch size)"
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output not in [0, 1] range (Sigmoid check failed)"

    # --------------------------------------------------------------------------
    # 5. Run Training Pipeline
    # --------------------------------------------------------------------------
    print(f"\nStep 5: Running Training Pipeline (Epochs={DEMO_EPOCHS})")
    # run_training encapsulates:
    # - DataLoader creation
    # - Model initialization
    # - Optimizer/Scheduler setup
    # - Training loop with Early Stopping
    # - Inference on Test Set
    # - Submission file generation
    run_training(epochs=DEMO_EPOCHS, batch_size=DEMO_BATCH_SIZE)

    # --------------------------------------------------------------------------
    # 6. Verify Submission File
    # --------------------------------------------------------------------------
    print("\nStep 6: Verifying Submission File")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    sub_df = pd.read_csv(submission_path)
    print(f"Submission Shape: {sub_df.shape}")
    print("First 5 rows:")
    print(sub_df.head())

    # Assertions for submission format
    expected_test_len = 100000  # Based on metadata/dataset analysis
    assert (
        len(sub_df) == expected_test_len
    ), f"Expected {expected_test_len} rows, got {len(sub_df)}"
    assert (
        "id" in sub_df.columns and "target" in sub_df.columns
    ), "Missing required columns 'id' or 'target'"
    assert (
        sub_df["target"].min() >= 0.0 and sub_df["target"].max() <= 1.0
    ), "Submission probabilities out of range [0, 1]"

    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
