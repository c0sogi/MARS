import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import (
    set_seed,
    setup_directories,
    TRAIN_CSV,
    CACHE_DIR,
    SUBMISSION_CSV,
    TARGET_COLS,
    DEVICE,
)
from library.utils import compute_kl_divergence, normalize_probabilities
from library.data_loader import SpectrogramDataset, compute_or_load_stats
from library.model import SpectrogramCRNN
from library.trainer import Trainer
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Code Demonstration ===")

    # 1. Setup
    set_seed(42)
    setup_directories()
    print("Setup complete.")

    # 2. Verify Utility Functions
    print("\n--- Verifying Utilities ---")

    # Test Normalization
    raw_preds = np.array([[0.1, 0.2, 0.7], [2.0, 3.0, 5.0], [0.0, 0.0, 0.0]])
    norm_preds = normalize_probabilities(raw_preds)

    # Check sums
    row_sums = norm_preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Normalization failed: Rows do not sum to 1.0"
    # Check zero handling (should be uniform)
    assert np.allclose(
        norm_preds[2], [1 / 3, 1 / 3, 1 / 3]
    ), "Normalization failed: Zero row not uniform"
    print("normalize_probabilities: OK")

    # Test KL Divergence
    # Case 1: Identical distributions (KL should be 0)
    y_true = np.array([[0.2, 0.3, 0.5]])
    y_pred_perfect = np.array([[0.2, 0.3, 0.5]])
    kl_perfect = compute_kl_divergence(y_true, y_pred_perfect)
    assert np.isclose(
        kl_perfect, 0.0, atol=1e-6
    ), f"KL Divergence failed: Expected 0.0, got {kl_perfect}"

    # Case 2: Different distributions
    y_pred_bad = np.array([[0.33, 0.33, 0.34]])
    kl_bad = compute_kl_divergence(y_true, y_pred_bad)
    assert (
        kl_bad > 0
    ), "KL Divergence failed: Metric should be positive for different distributions"
    print("compute_kl_divergence: OK")

    # 3. Data Loading (Optimized for Speed)
    print("\n--- Verifying Data Pipeline ---")

    # Load metadata
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_CSV}")

    full_train_df = pd.read_csv(TRAIN_CSV)

    # Sample a tiny subset to speed up the demo (Batch Size * 2)
    mini_batch_size = 32
    mini_df = full_train_df.sample(n=mini_batch_size * 2, random_state=42).reset_index(
        drop=True
    )
    print(f"Created mini dataset with {len(mini_df)} samples.")

    # Compute stats (uses cache if available, or computes on sample)
    mean, std = compute_or_load_stats(mini_df, load_cached_data=True)

    # Create Dataset
    dataset = SpectrogramDataset(mini_df, mode="train", mean=mean, std=std)

    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=mini_batch_size, shuffle=False, drop_last=True
    )

    # Fetch one batch
    data_batch, target_batch = next(iter(loader))

    # Assert Shapes
    # Expected: (Batch, Channels=4, Time=300, Freq=100)
    print(f"Input Batch Shape: {data_batch.shape}")
    print(f"Target Batch Shape: {target_batch.shape}")

    assert data_batch.shape == (mini_batch_size, 4, 300, 100), "Incorrect Input Shape"
    assert target_batch.shape == (mini_batch_size, 6), "Incorrect Target Shape"
    print("Data Pipeline: OK")

    # 4. Model Initialization & Forward Pass
    print("\n--- Verifying Model ---")
    model = SpectrogramCRNN()
    model.to(DEVICE)

    # Move batch to device
    data_batch = data_batch.to(DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(data_batch)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (mini_batch_size, 6), "Model output shape mismatch"

    # Check Softmax (Sum to 1)
    out_sums = output.sum(dim=1).cpu().numpy()
    assert np.allclose(
        out_sums, 1.0, atol=1e-4
    ), "Model output is not a valid probability distribution"
    print("Model Architecture: OK")

    # 5. Training Loop (Fast Demo)
    print("\n--- Verifying Training Loop ---")

    # Initialize Trainer with the mini loader for both train and val
    trainer = Trainer(model, loader, loader, device=DEVICE)

    # Run 1 epoch
    print("Running 1 epoch of training...")
    trainer.fit(epochs=1, lr=1e-3)

    # Check if model was saved
    saved_model_path = os.path.join(CACHE_DIR, "best_model.pth")
    assert os.path.exists(saved_model_path), "Model checkpoint was not saved."
    print("Training Loop: OK")

    # 6. Inference & Submission
    print("\n--- Verifying Inference ---")

    # Run inference using the model we just trained
    # This will load the test set defined in metadata/test.csv
    # Note: This runs on the full test set (9850 samples), but on A100/GPU this is fast.
    predict_and_submit(model_path=saved_model_path, device=DEVICE, batch_size=64)

    # Verify Submission File
    assert os.path.exists(SUBMISSION_CSV), "Submission file was not generated."

    sub_df = pd.read_csv(SUBMISSION_CSV)
    print(f"Submission Shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["eeg_id"] + TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # Check row sums
    vote_cols = sub_df[TARGET_COLS].values
    sums = vote_cols.sum(axis=1)
    assert np.allclose(
        sums, 1.0, atol=1e-4
    ), "Submission probabilities do not sum to 1.0"

    print("Inference: OK")
    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
