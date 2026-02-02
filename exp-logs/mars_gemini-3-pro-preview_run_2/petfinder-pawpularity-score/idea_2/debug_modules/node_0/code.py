import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from provided library files
from library.utils import seed_everything
from library.dataset import PawpularityDataset, get_transforms
from library.model import PawpularitySwinModel
from library.trainer import run_training, predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Pawpularity Pipeline Demonstration...")

    # 1. Setup and Configuration
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    # Define paths
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "validation.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    working_dir = "./working/demo_execution"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # 2. Dataset Class Demonstration
    print("\n[1/4] Verifying Dataset Class...")

    # Load a small sample of metadata to test the dataset class
    df_sample = pd.read_csv(train_csv).head(10)

    # Instantiate dataset with training transforms
    dataset = PawpularityDataset(df_sample, transforms=get_transforms("train"))

    # Fetch one item
    img, meta, target = dataset[0]

    # Verify Image
    assert isinstance(img, torch.Tensor), "Image must be a torch Tensor"
    assert img.shape == (
        3,
        224,
        224,
    ), f"Expected image shape (3, 224, 224), got {img.shape}"
    assert img.dtype == torch.float32, "Image tensor should be float32"

    # Verify Metadata
    # There are 12 binary features in the metadata
    assert isinstance(meta, np.ndarray), "Metadata features must be a numpy array"
    assert meta.shape == (12,), f"Expected metadata shape (12,), got {meta.shape}"

    # Verify Target
    # Target should be normalized to [0, 1] in the dataset class
    assert isinstance(target, (float, np.float32)), "Target must be a float"
    assert 0.0 <= target <= 1.0, f"Target {target} is out of normalized range [0, 1]"

    print("Dataset verification successful.")

    # 3. Model Architecture Demonstration
    print("\n[2/4] Verifying Model Architecture...")

    # Instantiate model
    # We use pretrained=False here just to check architecture quickly without downloading weights
    model = PawpularitySwinModel(pretrained=False)
    model.to(device)
    model.eval()

    # Create dummy inputs
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_meta = torch.randn(batch_size, 12).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images, dummy_meta)

    # Verify output shape (Batch_Size, 1)
    assert outputs.shape == (
        batch_size,
        1,
    ), f"Expected output shape ({batch_size}, 1), got {outputs.shape}"

    print("Model architecture verification successful.")

    # 4. Training Loop Demonstration
    print("\n[3/4] Running Training Loop (Debug Mode)...")

    # We use the 'debug=True' flag provided in library/trainer.py.
    # This automatically slices the dataframe to a small subset (100 train, 50 val).
    # We sets epochs=1 to ensure it finishes very quickly.
    model_save_path = run_training(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        output_dir=working_dir,
        epochs=1,
        batch_size=8,
        learning_rate_backbone=1e-5,
        learning_rate_head=1e-4,
        patience=1,
        debug=True,
        device=device,
    )

    # Verify the model file was created
    assert os.path.exists(
        model_save_path
    ), f"Best model file not found at {model_save_path}"
    print(f"Training complete. Model saved to: {model_save_path}")

    # 5. Prediction Demonstration
    print("\n[4/4] Generating Predictions...")

    submission_file = os.path.join(working_dir, "submission.csv")

    # Run inference using the model trained above
    predict(
        model_path=model_save_path,
        test_csv_path=test_csv,
        submission_path=submission_file,
        batch_size=32,
        device=device,
    )

    # Verify submission file
    assert os.path.exists(submission_file), "Submission file was not generated"

    df_sub = pd.read_csv(submission_file)

    # Check shape and columns
    assert "Id" in df_sub.columns, "Submission missing 'Id' column"
    assert "Pawpularity" in df_sub.columns, "Submission missing 'Pawpularity' column"
    assert (
        len(df_sub) == 992
    ), f"Expected 992 predictions (test set size), got {len(df_sub)}"

    # Check value ranges (Pawpularity should be between 1 and 100)
    min_score = df_sub["Pawpularity"].min()
    max_score = df_sub["Pawpularity"].max()

    assert min_score >= 1.0, f"Found predictions below 1.0: {min_score}"
    assert max_score <= 100.0, f"Found predictions above 100.0: {max_score}"

    print(f"Prediction successful. Submission saved to: {submission_file}")
    print(f"Prediction stats - Min: {min_score:.2f}, Max: {max_score:.2f}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
