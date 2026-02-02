import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure we can import from the current directory
sys.path.append(".")

from library import config
from library.data_utils import process_data, ManufacturingDataset, set_seed
from library.model import FEPFEModel
from library.train_eval import run_training


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demonstration on device: {device}")

    # 2. Data Processing
    print("\n[Demo] Processing Data...")
    # This will process the full dataset and save it to CACHE_DIR
    train_df, val_df, test_df, vocab_sizes, cont_cols, cat_cols = process_data(
        load_cached_data=True
    )

    # Verify Data Processing Outputs
    print("Verifying processed data...")
    assert not train_df.empty, "Processed training DataFrame is empty."
    assert not val_df.empty, "Processed validation DataFrame is empty."
    assert not test_df.empty, "Processed test DataFrame is empty."
    assert len(vocab_sizes) == 10, f"Expected 10 vocab sizes, got {len(vocab_sizes)}"
    assert len(cat_cols) == 10, "Expected 10 categorical columns."
    assert len(cont_cols) > 0, "Continuous columns list is empty."
    print("Data processing verification passed.")

    # 3. Optimize for Speed: Subsample Train/Val and Overwrite Cache
    # We subsample train and val to make the epoch run very fast.
    # We do NOT subsample test, so that the submission file is valid.
    print("\n[Demo] Optimizing for speed: Subsampling training data in cache...")

    # Sample 5% of data
    train_df_small = train_df.sample(frac=0.05, random_state=42)
    val_df_small = val_df.sample(frac=0.05, random_state=42)

    # Define cache paths (matching those in data_utils.py)
    cache_train_path = os.path.join(config.CACHE_DIR, "train_processed.parquet")
    cache_val_path = os.path.join(config.CACHE_DIR, "val_processed.parquet")

    # Overwrite cache
    train_df_small.to_parquet(cache_train_path, index=False)
    val_df_small.to_parquet(cache_val_path, index=False)

    print(f"Subsampled Train shape: {train_df_small.shape}")
    print(f"Subsampled Val shape: {val_df_small.shape}")
    print("Cache updated with subsampled data.")

    # 4. Verify Dataset Class
    print("\n[Demo] Verifying ManufacturingDataset...")
    # Use a tiny slice for quick verification
    dataset = ManufacturingDataset(
        train_df_small.iloc[:10], cont_cols, cat_cols, target_col="target"
    )
    sample = dataset[0]

    assert "continuous" in sample
    assert "categorical" in sample
    assert "target" in sample
    assert "id" in sample
    assert torch.is_tensor(sample["continuous"])
    assert torch.is_tensor(sample["categorical"])
    assert sample["continuous"].shape[0] == len(cont_cols)
    assert sample["categorical"].shape[0] == 10
    print("Dataset class verification passed.")

    # 5. Verify Model Architecture
    print("\n[Demo] Verifying FEPFEModel...")
    model = FEPFEModel(vocab_sizes=vocab_sizes, num_continuous=len(cont_cols))
    model.to(device)
    model.eval()

    # Create a dummy batch
    batch_size = 4
    dummy_cont = torch.randn(batch_size, len(cont_cols)).to(device)
    dummy_cat = torch.randint(0, 2, (batch_size, 10)).to(device)  # indices 0 or 1

    with torch.no_grad():
        # Forward pass
        stream_outputs = model(dummy_cont, dummy_cat)

    assert isinstance(stream_outputs, list), "Model output must be a list."
    assert (
        len(stream_outputs) == 5
    ), f"Expected 5 stream outputs, got {len(stream_outputs)}."
    for i, out in enumerate(stream_outputs):
        assert out.shape == (
            batch_size,
            1,
        ), f"Stream {i} output shape mismatch. Expected {(batch_size, 1)}, got {out.shape}"
    print("Model architecture verification passed.")

    # 6. Run Training Loop
    print("\n[Demo] Executing Training Loop (1 Epoch)...")
    # run_training will load the cached (subsampled) data.
    # We use a large batch size for speed.
    run_training(epochs=1, batch_size=4096, load_cached_data=True, save_model=True)

    # 7. Verify Submission
    print("\n[Demo] Verifying Submission...")
    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission shape: {sub_df.shape}")

        # Check against original test set length (which we didn't subsample)
        assert len(sub_df) == len(
            test_df
        ), f"Submission row count mismatch. Expected {len(test_df)}, got {len(sub_df)}."
        assert "id" in sub_df.columns
        assert "target" in sub_df.columns
        print("Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
