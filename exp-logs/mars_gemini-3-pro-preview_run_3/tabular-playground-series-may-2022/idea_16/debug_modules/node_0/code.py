import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.model import NoiseRegularizedFunnelMLP
from library.train_eval import train_model, predict, generate_submission, set_seed


def main():
    print("Starting demonstration of the Manufacturing Control pipeline...")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config parameters for a fast demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128
    Config.SUBMISSION_PATH = "./submission/demo_submission.csv"

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"    Device: {Config.DEVICE}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[2] Loading and preprocessing data (subset)...")

    # We use max_samples to limit the dataset size for speed
    demo_samples = 2048  # Enough for a few batches

    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Utilize existing cache if available
        max_samples=demo_samples,
    )

    # Verify DataLoaders
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Val loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    required_keys = ["cat_features", "cont_features", "target"]
    for key in required_keys:
        assert key in sample_batch, f"Batch missing key: {key}"

    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")
    print(f"    Test batches:  {len(test_loader)}")
    print(f"    Vocab sizes:   {metadata['vocab_sizes']}")
    print(f"    Continuous features: {len(metadata['cont_cols'])}")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[3] Initializing NoiseRegularizedFunnelMLP...")

    vocab_sizes = metadata["vocab_sizes"]
    num_cont_features = len(metadata["cont_cols"])

    # Instantiate model with smaller capacity for demo speed
    model = NoiseRegularizedFunnelMLP(
        vocab_sizes=vocab_sizes,
        num_cont_features=num_cont_features,
        embed_dim=8,
        hidden_layers=[64, 32],
        dropout_rate=0.1,
    ).to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        cat_x = sample_batch["cat_features"].to(Config.DEVICE)
        cont_x = sample_batch["cont_features"].to(Config.DEVICE)
        logits = model(cat_x, cont_x)

    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("    Model initialized and forward pass verified.")

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("\n[4] Running training loop...")

    # Train the model using the library function
    # This handles the optimizer, scheduler, and early stopping logic
    trained_model = train_model(
        train_loader, val_loader, metadata, epochs=Config.EPOCHS
    )

    print("    Training execution completed.")

    # --------------------------------------------------------------------------
    # 5. Prediction & Submission
    # --------------------------------------------------------------------------
    print("\n[5] Generating predictions and submission...")

    # Generate predictions on the test set (subset)
    predictions = predict(trained_model, test_loader, Config.DEVICE)

    print(f"    Generated {len(predictions)} predictions.")

    # Verify prediction range
    assert (predictions >= 0.0).all() and (
        predictions <= 1.0
    ).all(), "Predictions contain values outside [0, 1]"

    # Create a dummy sample submission file that matches the subset size
    # The library's generate_submission function expects the sample file to match the prediction length
    dummy_sub_path = os.path.join(Config.WORKING_DIR, "demo_sample_submission.csv")
    dummy_ids = np.arange(900000, 900000 + len(predictions))
    dummy_df = pd.DataFrame({"id": dummy_ids, "target": [0.5] * len(predictions)})
    dummy_df.to_csv(dummy_sub_path, index=False)

    # Generate the final submission file using the library function
    generate_submission(predictions, dummy_sub_path, Config.SUBMISSION_PATH)

    # Verify final output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    result_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(result_df) == len(predictions), "Submission file length mismatch."
    assert "target" in result_df.columns, "Submission file missing target column."

    print(f"    Submission successfully saved to {Config.SUBMISSION_PATH}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
