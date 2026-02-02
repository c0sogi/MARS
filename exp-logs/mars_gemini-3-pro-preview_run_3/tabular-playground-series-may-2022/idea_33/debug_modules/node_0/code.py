import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.dataset import get_datasets, ManufacturingDataset
from library.model import CRHPEModel
from library.trainer import run_training, Trainer


def main():
    # 1. Setup and Initialization
    print("=== Setting up environment ===")
    seed_everything(42)

    # Define directories
    base_metadata_dir = "./metadata"
    working_dir = "./working"
    cache_dir = os.path.join(working_dir, "demo_cache")
    submission_dir = "./submission"

    # Clean up previous demo runs if any
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)

    print(f"Device: {get_device()}")

    # 2. Demonstrate Data Loading (Debug Mode)
    print("\n=== Demonstrating Data Loading (Debug Mode) ===")
    # We use debug=True to load only 1000 samples for speed
    train_ds, val_ds, test_ds, vocab_sizes, test_ids = get_datasets(
        load_cached_data=False,  # Force processing from scratch for demo
        base_dir=base_metadata_dir,
        cache_dir=cache_dir,
        debug=True,
    )

    # Verify dataset sizes
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    # Assertions to ensure data loading logic is correct
    assert len(train_ds) == 1000, "Debug mode should limit train set to 1000 samples"
    assert len(val_ds) == 1000, "Debug mode should limit val set to 1000 samples"
    assert len(test_ds) == 1000, "Debug mode should limit test set to 1000 samples"
    assert len(vocab_sizes) > 0, "Vocab sizes should not be empty"

    # Inspect a single sample
    cat_sample, cont_sample, target_sample = train_ds[0]
    print(
        f"Sample 0 - Cat Shape: {cat_sample.shape}, Cont Shape: {cont_sample.shape}, Target: {target_sample}"
    )
    assert isinstance(cat_sample, torch.Tensor)
    assert isinstance(cont_sample, torch.Tensor)

    # 3. Demonstrate Model Instantiation and Forward Pass
    print("\n=== Demonstrating Model Architecture ===")
    # Determine number of continuous features from the dataset
    num_cont_features = train_ds.cont_features.shape[1]

    # Instantiate the model
    model = CRHPEModel(vocab_sizes=vocab_sizes, num_cont=num_cont_features)
    model.to(get_device())
    model.eval()

    # Create a dummy batch for verification
    batch_size = 8
    dummy_cat = torch.randint(0, 5, (batch_size, len(vocab_sizes))).to(get_device())
    dummy_cont = torch.randn(batch_size, num_cont_features).to(get_device())

    # Run forward pass
    with torch.no_grad():
        outputs = model(dummy_cat, dummy_cont)

    # The model returns a list of outputs (one for each stream)
    print(f"Number of output streams: {len(outputs)}")

    # Verify output structure
    assert len(outputs) == 5, "CRHPEModel should return outputs from 5 streams"
    for i, out in enumerate(outputs):
        assert out.shape == (
            batch_size,
            1,
        ), f"Stream {i} output shape mismatch. Expected {(batch_size, 1)}, got {out.shape}"

    print("Model forward pass verified successfully.")

    # 4. Demonstrate Full Training Pipeline
    print("\n=== Executing Full Training Pipeline (Fast Run) ===")
    # We use run_training from library.trainer which encapsulates the whole loop
    # Setting epochs=1 and debug=True ensures this completes very quickly
    run_training(
        epochs=1,
        batch_size=128,
        load_cached_data=True,  # Use the cache we just generated in step 2
        base_dir=base_metadata_dir,
        cache_dir=cache_dir,
        debug=True,
    )

    # 5. Verify Outputs
    print("\n=== Verifying Submission Outputs ===")
    submission_path = os.path.join(submission_dir, "submission.csv")
    model_path = "./working/best_model.pth"

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load submission and check content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print("Head of submission:")
    print(sub_df.head())

    # In debug mode, we expect 1000 predictions
    assert (
        len(sub_df) == 1000
    ), f"Expected 1000 predictions in debug mode, found {len(sub_df)}"
    assert (
        "id" in sub_df.columns and "target" in sub_df.columns
    ), "Submission columns mismatch"

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
