import os
import torch
import pandas as pd
import shutil
from library.utils import set_seed, get_device
from library.dataset import create_dataloaders
from library.model import FineTunedResNet18
from library.engine import run


def main():
    # 1. Setup and Configuration
    print("Initializing configuration...")
    set_seed(42)
    device = get_device()
    print(f"Computation device: {device}")

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Clean up working directory if it exists to ensure a fresh run
    if os.path.exists(SUBMISSION_DIR):
        shutil.rmtree(SUBMISSION_DIR)

    # 2. Demonstrate Dataset Loading
    print("\n=== Demonstrating Dataset Loading ===")
    # Create dataloaders with a small subset for verification speed
    batch_size = 8
    max_samples = 50

    dataloaders = create_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for simple debugging/demo to avoid overhead
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        max_samples=max_samples,
    )

    # Validate Dictionary Keys
    assert "train" in dataloaders, "Train loader missing"
    assert "val" in dataloaders, "Val loader missing"
    assert "test" in dataloaders, "Test loader missing"
    print("DataLoaders dictionary structure verified.")

    # Validate Training Batch
    train_loader = dataloaders["train"]
    images, labels = next(iter(train_loader))

    # Check Image Tensor Shape: [Batch, Channels, Height, Width]
    assert images.dim() == 4, "Image tensor should be 4D"
    assert (
        images.shape[0] == batch_size
    ), f"Batch size mismatch. Expected {batch_size}, got {images.shape[0]}"
    assert images.shape[1] == 3, "Images should be RGB (3 channels)"
    assert (
        images.shape[2] == 224 and images.shape[3] == 224
    ), "Images should be resized to 224x224"

    # Check Label Tensor Shape: [Batch]
    assert labels.dim() == 1, "Labels should be 1D"
    assert labels.shape[0] == batch_size, "Label batch size mismatch"
    assert labels.dtype == torch.float32, "Labels should be float32 for BCE loss"

    print(
        f"Train batch verified. Image shape: {images.shape}, Label shape: {labels.shape}"
    )

    # Validate Test Batch (Should return IDs instead of labels)
    test_loader = dataloaders["test"]
    t_images, t_ids = next(iter(test_loader))

    assert t_ids.dtype == torch.long, "Test IDs should be LongTensor"
    print(f"Test batch verified. IDs shape: {t_ids.shape}")

    # 3. Demonstrate Model Architecture
    print("\n=== Demonstrating Model Architecture ===")
    model = FineTunedResNet18()
    model.to(device)

    # Perform a forward pass
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    # Check Output Shape: [Batch, 1] (Logits)
    assert outputs.dim() == 2, "Model output should be 2D [Batch, 1]"
    assert outputs.shape[0] == batch_size, "Output batch size mismatch"
    assert outputs.shape[1] == 1, "Output should have 1 feature (logit)"

    print(f"Model forward pass successful. Output shape: {outputs.shape}")

    # 4. Demonstrate Full Engine Execution (Train + Predict)
    print("\n=== Demonstrating Engine Execution ===")
    # Run a minimal training loop: 1 epoch, very few samples
    # This verifies the integration of dataset, model, training loop, and submission generation

    run(
        num_epochs=1,
        batch_size=4,
        lr=1e-4,
        patience=1,
        max_samples=20,  # Limit samples to ensure this runs in seconds
        output_dir=SUBMISSION_DIR,
    )

    # 5. Validate Submission File
    print("\n=== Validating Submission Output ===")
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df = pd.read_csv(submission_path)
    print("Submission file loaded successfully.")
    print(df.head())

    # Check Columns
    assert "id" in df.columns, "Column 'id' missing in submission"
    assert "label" in df.columns, "Column 'label' missing in submission"

    # Check Data Integrity
    # Since we used max_samples=20 in run(), the test set was also sliced to 20
    assert len(df) == 20, f"Expected 20 predictions, found {len(df)}"

    # Check Probability Range
    assert df["label"].min() >= 0.0, "Probabilities cannot be negative"
    assert df["label"].max() <= 1.0, "Probabilities cannot exceed 1.0"

    print("Submission format and values verified.")
    print("\nAll demonstrations and validations completed successfully.")


if __name__ == "__main__":
    main()
