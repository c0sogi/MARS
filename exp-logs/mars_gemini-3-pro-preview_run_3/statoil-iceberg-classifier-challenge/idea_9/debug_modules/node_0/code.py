import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_loader import get_loaders, load_and_process_data
from library.model import SRNResNet
from library.train import train_one_epoch, validate, predict


def run_demo():
    print("Initializing Demo...")

    # 1. Configure for Speed and Demo Isolation
    # We modify the Config class attributes directly to run a fast check
    Config.EPOCHS = 2
    Config.DEBUG = True  # Forces data loader to use a small subset (e.g., 64 samples)
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"

    # Re-run setup to create the new working directory
    Config.setup()

    print(
        f"Configuration set: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # Set reproducible seed
    set_seed(Config.SEED)

    # 2. Demonstrate Data Loading
    print("\n--- Testing Data Pipeline ---")
    # This will load metadata, process images (or load cache), and return DataLoaders
    # Since DEBUG=True, these loaders will contain very few samples.
    train_loader, val_loader, test_loader, ids_test = get_loaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Set to 0 for simple main-thread debugging
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Verify Train Loader Batch
    images, angles, labels = next(iter(train_loader))

    print(
        f"Train Batch Shapes - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions for Data Integrity
    # Images: (B, 3, 75, 75) -> 3 channels (HH, HV, Avg)
    assert images.dim() == 4, "Image tensor must be 4D (B, C, H, W)"
    assert images.shape[1] == 3, "Image tensor must have 3 channels"
    assert (
        images.shape[2] == 75 and images.shape[3] == 75
    ), "Image dimensions must be 75x75"
    assert angles.dim() == 1, "Angles must be a 1D tensor"
    assert labels.dim() == 1, "Labels must be a 1D tensor"
    assert (
        images.shape[0] == angles.shape[0] == labels.shape[0]
    ), "Batch size mismatch across inputs"

    print("Data Pipeline Verification Passed.")

    # 3. Demonstrate Model Architecture
    print("\n--- Testing Model Architecture ---")
    model = SRNResNet().to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    angles = angles.to(Config.DEVICE)

    # Forward Pass
    outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions for Model
    assert outputs.dim() == 2, "Model output should be 2D (B, 1)"
    assert outputs.shape[0] == images.shape[0], "Output batch size does not match input"
    assert outputs.shape[1] == 1, "Output must be a single logit per sample"

    print("Model Architecture Verification Passed.")

    # 4. Demonstrate Training Loop Components
    print("\n--- Testing Training Loop ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run Training for Config.EPOCHS
    initial_loss = None
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        if initial_loss is None:
            initial_loss = train_loss

        # Basic sanity check: Loss should be a valid float
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    print("Training Loop Verification Passed.")

    # 5. Demonstrate Checkpointing
    print("\n--- Testing Checkpointing ---")
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_checkpoint.pth")

    # Save
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": Config.EPOCHS,
        },
        checkpoint_path,
    )

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"

    # Load
    new_model = SRNResNet().to(Config.DEVICE)
    checkpoint = load_checkpoint(
        checkpoint_path, new_model, optimizer=None, device=Config.DEVICE
    )

    # Verify weights match
    original_param = next(model.parameters())
    loaded_param = next(new_model.parameters())
    assert torch.equal(
        original_param, loaded_param
    ), "Model weights did not restore correctly"
    assert checkpoint["epoch"] == Config.EPOCHS, "Checkpoint metadata mismatch"

    print("Checkpointing Verification Passed.")

    # 6. Demonstrate Inference
    print("\n--- Testing Inference ---")
    # Use the predict function from library.train
    preds = predict(new_model, test_loader, Config.DEVICE)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Sample Predictions: {preds[:5].flatten()}")

    # Assertions
    # Since we used DEBUG=True, the test loader is a subset.
    # However, ids_test comes from the full metadata unless we manually sliced it in the caller.
    # The get_loaders(debug=True) slices the dataset, so len(test_loader.dataset) is small.
    expected_len = len(test_loader.dataset)
    assert (
        len(preds) == expected_len
    ), f"Prediction count {len(preds)} mismatch with test subset size {expected_len}"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities between 0 and 1"

    print("Inference Verification Passed.")

    # 7. Generate Dummy Submission
    print("\n--- Generating Demo Submission ---")
    # Note: ids_test in get_loaders returns the full list of IDs.
    # Since we ran inference on a subset, we just match the first N ids.
    subset_ids = ids_test[: len(preds)]

    submission_df = pd.DataFrame({"id": subset_ids, "is_iceberg": preds.flatten()})

    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
