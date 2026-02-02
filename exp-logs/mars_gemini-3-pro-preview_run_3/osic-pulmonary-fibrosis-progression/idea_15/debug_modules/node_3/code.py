import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, TargetScaler, LaplaceLogLikelihood
from library.data import get_dataloaders, LungDataset
from library.model import EADSNet
from library.train import run_training, train_one_epoch, evaluate


def setup_demo_environment():
    """
    Creates a lightweight environment for the demo by subsetting the data.
    This ensures the code runs quickly within the time limit.
    """
    print(">>> Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo"
    demo_metadata_dir = os.path.join(demo_dir, "metadata")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    demo_submission_dir = os.path.join(demo_dir, "submission")

    for d in [
        demo_metadata_dir,
        demo_cache_dir,
        demo_checkpoint_dir,
        demo_submission_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # 1. Subset Metadata
    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Select a few patients for the demo
    train_patients = train_df["Patient"].unique()[:3]
    val_patients = val_df["Patient"].unique()[:2]
    test_patients = test_df["Patient"].unique()[:2]

    demo_train = train_df[train_df["Patient"].isin(train_patients)].copy()
    demo_val = val_df[val_df["Patient"].isin(val_patients)].copy()
    demo_test = test_df[test_df["Patient"].isin(test_patients)].copy()

    # Save demo metadata
    demo_train_path = os.path.join(demo_metadata_dir, "train.csv")
    demo_val_path = os.path.join(demo_metadata_dir, "val.csv")
    demo_test_path = os.path.join(demo_metadata_dir, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # 2. Create Dummy Sample Submission
    # The sample submission needs to match the test patients
    # We create predictions for weeks -12 to 133 for these patients
    submission_rows = []
    for patient in test_patients:
        for week in range(-12, 14):  # Short range for demo
            submission_rows.append(
                {"Patient_Week": f"{patient}_{week}", "FVC": 2000, "Confidence": 100}
            )

    demo_sample_sub = pd.DataFrame(submission_rows)
    demo_sample_sub_path = os.path.join(demo_dir, "sample_submission.csv")
    demo_sample_sub.to_csv(demo_sample_sub_path, index=False)

    # 3. Monkey-patch Config
    # We modify the Config class directly to point to our demo files and settings
    Config.TRAIN_CSV = demo_train_path
    Config.VAL_CSV = demo_val_path
    Config.TEST_CSV = demo_test_path
    Config.SAMPLE_SUBMISSION = demo_sample_sub_path

    Config.CACHE_DIR = demo_cache_dir
    Config.CHECKPOINT_DIR = demo_checkpoint_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.BEST_MODEL_PATH = os.path.join(demo_checkpoint_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    # Reduce compute load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.PATIENCE = 1

    print(">>> Demo environment configured.")


def test_utils():
    """
    Verifies the logic of utility functions.
    """
    print("\n>>> Testing Utilities...")

    # 1. Test TargetScaler
    scaler = TargetScaler()
    dummy_data = np.array([1000, 2000, 3000], dtype=np.float32)
    scaler.fit(dummy_data)

    assert scaler.mean == 2000.0, f"Scaler mean incorrect: {scaler.mean}"
    assert np.isclose(scaler.std, 816.4966), f"Scaler std incorrect: {scaler.std}"

    transformed = scaler.transform(dummy_data)
    expected_transformed = (dummy_data - 2000.0) / scaler.std
    assert np.allclose(transformed, expected_transformed), "Scaler transform failed"

    inverse = scaler.inverse_transform(transformed)
    assert np.allclose(inverse, dummy_data), "Scaler inverse_transform failed"

    # Test sigma inverse (should only scale by std, not add mean)
    sigma_scaled = np.array([1.0], dtype=np.float32)
    sigma_orig = scaler.inverse_transform_sigma(sigma_scaled)
    assert np.isclose(sigma_orig, scaler.std), "Scaler inverse_transform_sigma failed"

    print("TargetScaler: OK")

    # 2. Test LaplaceLogLikelihood
    # Case 1: Perfect prediction, sigma clipped at 70
    y_true = torch.tensor([2500.0])
    y_pred = torch.tensor([2500.0])
    sigma = torch.tensor([10.0])  # Should be clipped to 70

    score = LaplaceLogLikelihood(y_true, y_pred, sigma)
    # Delta = 0
    # Sigma_clipped = 70
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    expected_score = -np.log(np.sqrt(2) * 70)
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Metric calculation failed. Got {score}, expected {expected_score}"

    print("LaplaceLogLikelihood: OK")


def test_data_pipeline():
    """
    Verifies data loading, image caching, and batch structure.
    """
    print("\n>>> Testing Data Pipeline...")

    # This will trigger cache generation for our subset
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=True
    )

    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = {"image", "tabular", "time", "target", "patient_week", "raw_fvc"}
    assert expected_keys.issubset(batch.keys()), f"Batch missing keys: {batch.keys()}"

    # Verify shapes
    # Image: (Batch, 3, H, W) -> (2, 3, 260, 260)
    images = batch["image"]
    assert images.dim() == 4, "Image tensor has wrong dimensions"
    assert images.shape[1] == 3, "Image tensor has wrong channel count (slices)"
    assert images.shape[2] == Config.IMG_SIZE, "Image tensor has wrong height"

    # Tabular: (Batch, 4)
    tabular = batch["tabular"]
    assert tabular.shape[1] == 4, "Tabular features have wrong dimension"

    # Time: (Batch, 1)
    time_rel = batch["time"]
    assert time_rel.shape[1] == 1, "Time feature has wrong dimension"

    # Target: (Batch, 1)
    target = batch["target"]
    assert target.shape[1] == 1, "Target has wrong dimension"

    print(f"Batch loaded successfully. Image shape: {images.shape}")
    return train_loader, val_loader, test_loader, scaler


def test_model_logic(train_loader):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n>>> Testing Model Logic...")

    device = torch.device(Config.DEVICE)
    model = EADSNet().to(device)

    # Get a batch
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    tabular = batch["tabular"].to(device)
    time_rel = batch["time"].to(device)

    # Forward pass
    mu, sigma = model(images, tabular, time_rel)

    # Check shapes
    assert mu.shape == (Config.BATCH_SIZE, 1), f"Mu shape mismatch: {mu.shape}"
    assert sigma.shape == (Config.BATCH_SIZE, 1), f"Sigma shape mismatch: {sigma.shape}"

    # Check sigma positivity (Softplus + Epsilon)
    assert torch.all(sigma > 0), "Sigma contains non-positive values"

    # Check if gradients flow
    loss = torch.mean((mu - 0) ** 2)
    loss.backward()

    # Check if backbone gradients are set correctly (some frozen, some unfrozen)
    # We expect the first layer of backbone to be frozen
    first_conv = model.image_encoder.backbone.conv_stem
    assert first_conv.weight.grad is None, "Backbone stem should be frozen"

    # We expect the head to be unfrozen
    head_linear = model.wide_stream
    assert head_linear.weight.grad is not None, "Wide stream head should have gradients"

    print("Model forward/backward pass: OK")
    return model


def test_training_execution():
    """
    Runs the full training loop (shortened) to verify integration.
    """
    print("\n>>> Testing Full Training Execution...")

    # We use the run_training function provided in library.train
    # Since we patched Config, it will use our demo data and settings
    try:
        run_training()
    except Exception as e:
        raise RuntimeError(f"Training execution failed: {e}")

    # Verify artifacts
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model checkpoint not found"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert len(sub_df) > 0

    # Check confidence clipping in submission
    min_conf = sub_df["Confidence"].min()
    assert min_conf >= 70, f"Submission contains confidence < 70: {min_conf}"

    print("Training execution: OK")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # 1. Setup
    setup_demo_environment()

    # 2. Verify Utils
    test_utils()

    # 3. Verify Data
    train_loader, val_loader, test_loader, scaler = test_data_pipeline()

    # 4. Verify Model
    model = test_model_logic(train_loader)

    # 5. Verify Training Loop
    test_training_execution()

    print("\n>>> All demonstrations completed successfully.")
