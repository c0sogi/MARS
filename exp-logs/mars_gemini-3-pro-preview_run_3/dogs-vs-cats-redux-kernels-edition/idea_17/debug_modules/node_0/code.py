import os
import torch
import torch.optim as optim
import pandas as pd
from library.utils import seed_everything, get_device
from library.models import get_model
from library.data import get_loaders, get_test_loader
from library.engine import (
    train_one_epoch,
    validate_one_epoch,
    predict_with_tta,
    save_submission,
)


def run_demonstration():
    print("Starting Library Demonstration...")

    # 1. Setup Environment
    # Set seeds for reproducibility
    seed_everything(42)
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Loading Demonstration
    print("\n--- 1. Testing Data Loading ---")
    # We use ResNet50 which requires 256x256 resolution
    # We set load_cached_data=False to ensure the fold creation logic runs
    train_loader, val_loader = get_loaders(
        fold_idx=0,
        model_name="resnet50",
        batch_size=4,
        num_workers=2,
        load_cached_data=False,
        seed=42,
    )

    # Fetch a single batch to verify shapes
    # Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        # Assertions
        assert images.shape == (
            4,
            3,
            256,
            256,
        ), f"Expected (4, 3, 256, 256), got {images.shape}"
        assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
        assert labels.dtype == torch.float32, "Labels should be float32"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Val Loader
    try:
        val_images, val_labels = next(iter(val_loader))
        print(f"Val Batch   - Images: {val_images.shape}")
        assert val_images.shape == (4, 3, 256, 256)
    except StopIteration:
        raise AssertionError("Val loader is empty!")

    # 3. Model Instantiation Demonstration
    print("\n--- 2. Testing Model Instantiation ---")
    # Using pretrained=False for speed and to avoid downloading weights in this demo
    model = get_model("resnet50", pretrained=False)
    model.to(device)

    # Verify forward pass with dummy input
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output (2, 1), got {output.shape}"

    # 4. Training Engine Demonstration
    print("\n--- 3. Testing Training & Validation Engine ---")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train for 1 epoch, limited to 2 batches for speed
    print("Running training step (limited to 2 batches)...")
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=1,
        max_batches=2,
    )

    print(f"Returned Train Loss: {train_loss}")
    assert isinstance(train_loss, float), "Train loss must be a float"
    # Loss might be high initially, but shouldn't be NaN or Inf usually
    assert train_loss >= 0, "Train loss should be non-negative"

    # Validate, limited to 2 batches
    print("Running validation step (limited to 2 batches)...")
    val_loss = validate_one_epoch(
        model=model, loader=val_loader, device=device, max_batches=2
    )
    print(f"Returned Val Loss: {val_loss}")
    assert isinstance(val_loss, float), "Val loss must be a float"

    # 5. Inference Demonstration
    print("\n--- 4. Testing Inference (TTA) ---")
    test_loader = get_test_loader(model_name="resnet50", batch_size=4, num_workers=2)

    # Predict with Test Time Augmentation, limited to 2 batches
    print("Running prediction (limited to 2 batches)...")
    preds_df = predict_with_tta(
        model=model, loader=test_loader, device=device, max_batches=2
    )

    print("Predictions Sample:")
    print(preds_df.head())

    # Verify Predictions DataFrame
    assert isinstance(preds_df, pd.DataFrame), "Result must be a DataFrame"
    assert "id" in preds_df.columns, "DataFrame must contain 'id' column"
    assert "label" in preds_df.columns, "DataFrame must contain 'label' column"
    assert len(preds_df) > 0, "Predictions DataFrame is empty"

    # Check probability range
    probs = preds_df["label"].values
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"

    # 6. Submission Saving
    print("\n--- 5. Testing Submission Saving ---")
    output_path = "./working/demo_submission/submission.csv"
    save_submission(preds_df, output_path)

    assert os.path.exists(
        output_path
    ), f"Submission file was not created at {output_path}"
    print(f"Submission successfully saved to {output_path}")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
