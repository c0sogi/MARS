import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.utils import seed_everything, get_device
from library.dataset import AppleDataset, get_transforms, TARGET_COLS
from library.model import ResNet18Baseline
from library.train import train_model
from library.inference import predict_and_submit


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Initializing demonstration...")
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"

    # Clean working directory if it exists to ensure a fresh run
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    # ==========================================
    # 2. Dataset and Transform Verification
    # ==========================================
    print("\n[1/4] Verifying Dataset and Transforms...")

    # Instantiate dataset with training transforms
    train_dataset = AppleDataset(
        metadata_path=train_meta_path,
        transform=get_transforms("train", image_size=256),
        input_dir=INPUT_DIR,
        mode="train",
    )

    # Validate length
    print(f"  Dataset length: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset should not be empty."

    # Validate single item retrieval
    image, label = train_dataset[0]

    # Check image tensor shape: (Channels, Height, Width)
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
    assert image.shape == (
        3,
        256,
        256,
    ), f"Expected image shape (3, 256, 256), got {image.shape}"

    # Check label type and shape
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
    assert label.ndim == 0, "Label should be a scalar tensor (0-dim)"

    print("  Dataset verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[2/4] Verifying Model Architecture...")

    num_classes = len(TARGET_COLS)
    model = ResNet18Baseline(num_classes=num_classes, pretrained=False)
    model.to(device)
    model.eval()

    # Create a dummy batch: (Batch_Size, Channels, Height, Width)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch_Size, Num_Classes)
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Expected output shape ({batch_size}, {num_classes}), got {output.shape}"

    print("  Model architecture verification passed.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n[3/4] Demonstrating Training Loop...")

    # We use a small subset (max_samples=50) and 2 epochs for speed
    model_save_dir = os.path.join(WORKING_DIR, "models")

    best_model_path = train_model(
        train_metadata_path=train_meta_path,
        val_metadata_path=val_meta_path,
        input_dir=INPUT_DIR,
        output_dir=model_save_dir,
        epochs=2,
        batch_size=8,
        learning_rate=1e-4,
        seed=42,
        patience=2,
        max_samples=50,  # Limit data for quick demonstration
    )

    # Verify model file creation
    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    print(f"  Training complete. Model saved to {best_model_path}")

    # ==========================================
    # 5. Inference and Submission Demonstration
    # ==========================================
    print("\n[4/4] Demonstrating Inference and Submission...")

    submission_path = os.path.join(WORKING_DIR, "submission.csv")

    predict_and_submit(
        model_path=best_model_path,
        test_metadata_path=test_meta_path,
        input_dir=INPUT_DIR,
        output_path=submission_path,
        batch_size=8,
        device=device,
    )

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["image_id"] + TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check that probabilities sum to roughly 1 (softmax applied)
    # Note: Due to floating point precision, we check closeness to 1
    prob_cols = df_sub[TARGET_COLS]
    row_sums = prob_cols.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("  Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
