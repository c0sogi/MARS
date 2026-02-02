import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import (
    setup_environment,
    SUBMISSION_FILE,
    MODEL_SAVE_PATH,
    DEVICE,
    SEED,
)
from library.data import DataProcessor
from library.model import DualStreamFunnelMLP
from library.engine import train_engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration of the Manufacturing Control Pipeline...")

    # 1. Setup Environment
    # Ensures directories exist and sets random seeds for reproducibility
    setup_environment(seed=SEED)

    # 2. Data Processing and Loading
    # We use debug=True and max_samples=2000 to ensure the demonstration runs quickly.
    # In a full run, set debug=False to use the entire dataset.
    print("\nInitializing DataProcessor...")
    processor = DataProcessor()

    # Note: load_cached_data=False forces reprocessing to demonstrate the logic,
    # but in practice, True is preferred to save time.
    train_loader, val_loader, test_loader, vocab_sizes = processor.get_dataloaders(
        load_cached_data=False, debug=True, max_samples=2000
    )

    # 3. Validate Data Loaders
    print("\nValidating Data Loaders...")
    try:
        batch = next(iter(train_loader))
        cat_x = batch["cat"]
        cont_x = batch["cont"]
        targets = batch["target"]

        print(
            f"Batch shapes - Cat: {cat_x.shape}, Cont: {cont_x.shape}, Target: {targets.shape}"
        )

        # Assertions to ensure data integrity
        assert cat_x.dim() == 2, "Categorical input should be 2D"
        assert cont_x.dim() == 2, "Continuous input should be 2D"
        assert targets.dim() == 2, "Target should be 2D (Batch, 1)"
        assert len(vocab_sizes) > 0, "Vocabulary sizes should not be empty"

        print("Data Loader validation passed.")
    except StopIteration:
        raise RuntimeError("Data loader is empty!")

    # 4. Model Initialization
    print("\nInitializing Model...")
    # Determine continuous dimension from the data
    cont_dim = cont_x.shape[1]

    model = DualStreamFunnelMLP(
        vocab_sizes=vocab_sizes,
        cont_dim=cont_dim,
        embed_dim=8,  # Reduced for demo
        backbone_layers=[64, 32],  # Reduced for demo
        dropout=0.1,
    )

    # Move model to device for verification
    model = model.to(DEVICE)

    # Verify forward pass
    with torch.no_grad():
        logits = model(cat_x.to(DEVICE), cont_x.to(DEVICE))
        assert logits.shape == (
            cat_x.shape[0],
            1,
        ), f"Output shape mismatch: {logits.shape}"
    print("Model initialization and forward pass verification passed.")

    # 5. Training and Evaluation
    print("\nStarting Training Engine...")
    # We run for a minimal number of epochs to demonstrate the loop
    train_engine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=2,
        max_lr=1e-3,
        weight_decay=1e-5,
        patience=1,
    )

    # 6. Submission Validation
    print("\nValidating Submission File...")
    if not os.path.exists(SUBMISSION_FILE):
        raise FileNotFoundError(f"Submission file was not created at {SUBMISSION_FILE}")

    df_sub = pd.read_csv(SUBMISSION_FILE)

    # Check columns
    expected_cols = ["id", "target"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check shape (should match test_loader size, which is limited by max_samples in debug mode)
    # Note: DataLoader drops the last batch if drop_last=True (only for train),
    # but test loader usually keeps all. However, batch_size might affect total count slightly if logic varies.
    # Here we just ensure it's not empty.
    assert len(df_sub) > 0, "Submission file is empty"

    # Check value ranges
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print(f"Submission file validated successfully. Shape: {df_sub.shape}")
    print("\nDemonstration complete.")


if __name__ == "__main__":
    main()
