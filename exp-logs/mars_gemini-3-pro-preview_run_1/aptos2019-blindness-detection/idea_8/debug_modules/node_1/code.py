import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import RetinopathyDataset
from library.model import RetinopathyModel
from library.engine import train_model


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Overrides for Speed and Demonstration
    # We modify the Config class attributes directly to control the pipeline behavior.
    print("Configuring parameters for rapid execution...")
    seed_everything(42)
    Config.epochs = 1  # Run only 1 epoch
    Config.debug = True  # Use debug mode (affects cache naming)
    Config.train_batch_size = 4  # Small batch size for the subset
    Config.val_batch_size = 4
    Config.pretrained = False  # Disable pretrained weights to avoid network calls
    Config.num_workers = 2  # Reduce workers for small data

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # 2. Data Preparation
    # Load metadata and subset heavily to ensure the script finishes quickly.
    print("Loading and subsetting metadata...")

    if not os.path.exists(Config.train_metadata_path):
        raise FileNotFoundError(f"Metadata not found at {Config.train_metadata_path}")

    # Use a tiny subset: 12 training samples, 8 validation, 8 test
    df_train = pd.read_csv(Config.train_metadata_path).head(12)
    df_val = pd.read_csv(Config.val_metadata_path).head(8)
    df_test = pd.read_csv(Config.test_metadata_path).head(8)

    print(
        f"Subset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
    )

    # 3. Dataset Instantiation and Verification
    print("Initializing Datasets...")
    # load_cached_data=False ensures we process only our small subset and don't look for full dataset caches
    train_dataset = RetinopathyDataset(df_train, phase="train", load_cached_data=False)
    val_dataset = RetinopathyDataset(df_val, phase="val", load_cached_data=False)
    test_dataset = RetinopathyDataset(df_test, phase="test", load_cached_data=False)

    print("Verifying Dataset logic...")
    # Check length
    assert len(train_dataset) == 12, "Train dataset length mismatch"

    # Check item structure
    img, target = train_dataset[0]
    # Image should be (3, H, W)
    assert img.shape == (
        3,
        Config.image_size,
        Config.image_size,
    ), f"Image shape mismatch: {img.shape}"
    # Target should be (num_ordinal_heads,) which is 4
    assert target.shape == (
        Config.num_ordinal_heads,
    ), f"Target shape mismatch: {target.shape}"
    assert target.dtype == torch.float32 or target.dtype == np.float32

    # Verify Ordinal Encoding Logic
    # Diagnosis 0 -> [0, 0, 0, 0]
    # Diagnosis 2 -> [1, 1, 0, 0]
    # Diagnosis 4 -> [1, 1, 1, 1]
    sample_diagnosis = df_train.iloc[0]["diagnosis"]
    expected_sum = float(sample_diagnosis)
    assert (
        target.sum() == expected_sum
    ), f"Ordinal encoding incorrect. Diagnosis: {sample_diagnosis}, Target: {target}"
    print("Dataset verification passed.")

    # 4. Model Instantiation and Verification
    print("Initializing Model...")
    model = RetinopathyModel()
    model.eval()

    # Check Forward Pass
    print("Verifying Model forward pass...")
    dummy_batch = torch.randn(2, 3, Config.image_size, Config.image_size)
    with torch.no_grad():
        outputs = model(dummy_batch)

    # Output shape should be (Batch_Size, Num_Ordinal_Heads) -> (2, 4)
    assert outputs.shape == (
        2,
        Config.num_ordinal_heads,
    ), f"Model output shape mismatch. Expected (2, 4), got {outputs.shape}"
    print("Model verification passed.")

    # 5. Metric Verification
    print("Verifying QWK Metric...")
    # Perfect agreement
    score_perfect = quadratic_weighted_kappa([0, 1, 4], [0, 1, 4])
    assert np.isclose(score_perfect, 1.0), f"QWK should be 1.0, got {score_perfect}"
    # Random/Bad agreement
    score_bad = quadratic_weighted_kappa([0, 0, 0], [4, 4, 4])
    # QWK can be 0 or negative
    assert score_bad < 0.5, f"QWK should be low for mismatch, got {score_bad}"
    print("Metric verification passed.")

    # 6. Training Pipeline Execution
    print("Starting Training Loop (using engine.train_model)...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    # Run the engine
    # This will train for 1 epoch, validate, and then run inference on test_loader
    train_model(train_loader, val_loader, test_loader, df_test)

    # 7. Output Verification
    print("Verifying output files...")
    submission_path = os.path.join(Config.working_dir, "submission.csv")
    model_path = os.path.join(Config.working_dir, "best_model.pth")

    assert os.path.exists(submission_path), "Submission file was not created."
    assert os.path.exists(model_path), "Best model file was not created."

    # Check submission content
    sub_df = pd.read_csv(submission_path)
    assert len(sub_df) == len(
        df_test
    ), f"Submission rows {len(sub_df)} != Test rows {len(df_test)}"
    assert list(sub_df.columns) == [
        "id_code",
        "diagnosis",
    ], f"Submission columns mismatch: {sub_df.columns}"

    print("=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
