import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    metric_laplace_log_likelihood,
    save_submission_file,
)
from library.data import get_dataloaders, LungDataset
from library.model import MACLINet
from library.train import Trainer


def create_subset_metadata(source_path, dest_path, n_patients=5):
    """Creates a smaller subset of the metadata for demonstration purposes."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Get unique patients and sample a subset
    patients = df["Patient"].unique()
    if len(patients) > n_patients:
        subset_patients = patients[:n_patients]
        df_subset = df[df["Patient"].isin(subset_patients)].reset_index(drop=True)
    else:
        df_subset = df.copy()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df_subset.to_csv(dest_path, index=False)
    print(
        f"Created subset {dest_path} with {len(df_subset)} rows ({df_subset['Patient'].nunique()} patients)."
    )
    return df_subset


def test_utils():
    print("\n=== Testing Library: utils.py ===")

    # Test Metric: Laplace Log Likelihood
    # Case 1: Perfect prediction with fixed confidence
    y_true = np.array([2000, 3000])
    y_pred = np.array([2000, 3000])
    sigma = np.array([100, 100])  # > 70, so not clipped

    # Formula: - (sqrt(2) * 0) / 100 - ln(sqrt(2) * 100)
    # = - ln(141.421356) ~= -4.9517
    score = metric_laplace_log_likelihood(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 100)

    print(f"Metric Score (Perfect Pred): {score:.4f}")
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), "Metric calculation mismatch for perfect prediction"

    # Case 2: Large error (clipped at 1000)
    y_true_err = np.array([2000])
    y_pred_err = np.array([4000])  # Delta = 2000 -> Clipped to 1000
    sigma_err = np.array([50])  # Clipped to 70

    score_err = metric_laplace_log_likelihood(y_true_err, y_pred_err, sigma_err)

    # Formula: - (sqrt(2) * 1000) / 70 - ln(sqrt(2) * 70)
    # = - (1414.21356 / 70) - ln(98.9949)
    # = - 20.203 - 4.595 ~= -24.798
    expected_score_err = -(np.sqrt(2) * 1000) / 70.0 - np.log(np.sqrt(2) * 70.0)

    print(f"Metric Score (Large Error): {score_err:.4f}")
    assert np.isclose(
        score_err, expected_score_err, atol=1e-4
    ), "Metric calculation mismatch for large error"

    # Test Save Submission
    dummy_sub = pd.DataFrame(
        {
            "Patient_Week": ["ID1_1", "ID1_2"],
            "FVC": [2000, 2000],
            "Confidence": [100, 100],
        }
    )
    dummy_path = os.path.join(Config.WORKING_DIR, "test_sub.csv")
    save_submission_file(dummy_sub, dummy_path)
    assert os.path.exists(dummy_path), "Submission file was not created"
    print("Utils verification passed.")


def test_data_and_model():
    print("\n=== Testing Library: data.py & model.py ===")

    # 1. Get DataLoaders
    # This will trigger preprocessing on the subset data
    train_loader, val_loader, test_loader, scalers = get_dataloaders(
        load_cached_data=True
    )

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    # 2. Inspect a batch
    batch = next(iter(train_loader))
    (images, clinical), targets = batch

    # Verify Shapes
    # Image: (B, 3, H, W) -> H,W depend on Config.IMG_SIZE (default 260)
    # Clinical: (B, 6)
    # Target: (B,)
    print(f"Image Batch Shape: {images.shape}")
    print(f"Clinical Batch Shape: {clinical.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    assert images.ndim == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (slices)"
    assert clinical.shape[1] == 6, "Clinical features should have 6 dimensions"

    # 3. Instantiate Model
    model = MACLINet()
    model.to(Config.DEVICE)

    # 4. Forward Pass
    images = images.to(Config.DEVICE)
    clinical = clinical.to(Config.DEVICE)

    with torch.no_grad():
        mu, sigma = model(images, clinical)

    print(f"Model Output Mu Shape: {mu.shape}")
    print(f"Model Output Sigma Shape: {sigma.shape}")

    assert mu.shape == targets.shape, "Output Mu shape mismatch"
    assert sigma.shape == targets.shape, "Output Sigma shape mismatch"
    assert (sigma > 0).all(), "Sigma must be positive"

    print("Data and Model verification passed.")
    return model, train_loader, val_loader, test_loader, scalers


def test_training(model, train_loader, val_loader, test_loader, scalers):
    print("\n=== Testing Library: train.py ===")

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, scalers)

    # Run 1 Epoch of Training
    print("Running training epoch...")
    loss = trainer.train_epoch(0)
    print(f"Train Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss returned NaN"

    # Run Validation
    print("Running validation...")
    val_score = trainer.validate()
    print(f"Validation Score: {val_score:.4f}")
    assert not np.isnan(val_score), "Validation score returned NaN"

    # Run Prediction (Submission Generation)
    # We need to save a dummy best model first because predict() loads it
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print("Running prediction...")
    trainer.predict()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns

    print("Training verification passed.")


def main():
    seed_everything(42)

    # --- 1. Configure for Demo Speed ---
    print("Configuring environment...")
    Config.EXPERIMENT_NAME = "demo_execution_custom"
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist (Config.setup() ran on import, but we changed paths)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Speed optimizations
    Config.EPOCHS = 1
    Config.PRETRAINED = False  # Avoid downloading weights
    Config.IMG_SIZE = 128  # Smaller images for faster resizing
    Config.BATCH_SIZE = 4

    # --- 2. Create Data Subsets ---
    # We create temporary CSVs with fewer patients to speed up preprocessing
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")
    subset_test_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")

    # Use existing metadata to create subsets
    create_subset_metadata("./metadata/train.csv", subset_train_path, n_patients=3)
    create_subset_metadata("./metadata/val.csv", subset_val_path, n_patients=2)
    create_subset_metadata("./metadata/test.csv", subset_test_path, n_patients=2)

    # Override Config paths
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path
    Config.TEST_CSV = subset_test_path
    # Note: SAMPLE_SUBMISSION is used in get_dataloaders for the test set structure.
    # We need to filter sample_submission to match our test subset patients.
    test_subset_df = pd.read_csv(subset_test_path)
    test_patients = test_subset_df["Patient"].unique()

    orig_sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    # Filter for patients in our subset
    subset_sample_sub = orig_sample_sub[
        orig_sample_sub["Patient_Week"].apply(
            lambda x: x.split("_")[0] in test_patients
        )
    ].copy()

    subset_sample_sub_path = os.path.join(
        Config.WORKING_DIR, "sample_submission_subset.csv"
    )
    subset_sample_sub.to_csv(subset_sample_sub_path, index=False)
    Config.SAMPLE_SUBMISSION = subset_sample_sub_path

    # --- 3. Run Tests ---
    try:
        test_utils()
        model, train_loader, val_loader, test_loader, scalers = test_data_and_model()
        test_training(model, train_loader, val_loader, test_loader, scalers)
        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nERROR during execution: {e}")
        raise e

    finally:
        # Cleanup temporary files (optional, but good practice)
        if os.path.exists(subset_train_path):
            os.remove(subset_train_path)
        if os.path.exists(subset_val_path):
            os.remove(subset_val_path)
        if os.path.exists(subset_test_path):
            os.remove(subset_test_path)
        if os.path.exists(subset_sample_sub_path):
            os.remove(subset_sample_sub_path)


if __name__ == "__main__":
    main()
