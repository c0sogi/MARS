import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import SLHDAN
from library.train import train_model, predict_and_submit

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=" * 50)
    print("STARTING DEMO SCRIPT")
    print("=" * 50)

    # 1. Setup Environment
    seed_everything(42)

    # Define paths for demo artifacts
    DEMO_DIR = "./working/demo_artifacts"
    DEMO_METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    DEMO_MODEL_DIR = os.path.join(DEMO_DIR, "models")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_MODEL_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    print("\n[1] Verifying Metric Logic...")
    # Test Case 1: Perfect prediction with standard confidence
    y_true = torch.tensor([2000.0])
    y_pred = torch.tensor([2000.0])
    sigma = torch.tensor([100.0])

    # Formula: - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
    # Delta = 0, Sigma_clipped = 100
    # Term 1 = 0
    # Term 2 = ln(141.421356) approx 4.9517
    # Metric approx -4.9517
    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    expected = -np.log(np.sqrt(2) * 100)

    print(f"   Score: {score.item():.4f}, Expected: {expected:.4f}")
    assert np.isclose(
        score.item(), expected, atol=1e-4
    ), "Metric calculation mismatch for perfect prediction"

    # Test Case 2: Large error (clipped) and low confidence (clipped)
    y_true = torch.tensor([2000.0])
    y_pred = torch.tensor([3500.0])  # Delta 1500 -> Clipped to 1000
    sigma = torch.tensor([10.0])  # Clipped to 70

    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)

    delta_clipped = 1000.0
    sigma_clipped = 70.0
    expected = -(np.sqrt(2) * delta_clipped) / sigma_clipped - np.log(
        np.sqrt(2) * sigma_clipped
    )

    print(f"   Score: {score.item():.4f}, Expected: {expected:.4f}")
    assert np.isclose(
        score.item(), expected, atol=1e-4
    ), "Metric calculation mismatch for clipped values"
    print("   Metric verification passed.")

    print("\n[2] Preparing Subset Data for Speed...")
    # Load original metadata
    orig_train = pd.read_csv(Config.TRAIN_CSV)
    orig_val = pd.read_csv(Config.VAL_CSV)
    orig_test = pd.read_csv(Config.TEST_CSV)

    # Sample subsets (ensure we have enough data for a tiny batch)
    # We group by Patient to ensure we don't split patient records
    train_patients = orig_train["Patient"].unique()[:5]
    val_patients = orig_val["Patient"].unique()[:2]
    test_patients = orig_test["Patient"].unique()[:2]

    sub_train = orig_train[orig_train["Patient"].isin(train_patients)].copy()
    sub_val = orig_val[orig_val["Patient"].isin(val_patients)].copy()
    sub_test = orig_test[orig_test["Patient"].isin(test_patients)].copy()

    # Save subsets
    demo_train_path = os.path.join(DEMO_METADATA_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_METADATA_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_METADATA_DIR, "test.csv")

    sub_train.to_csv(demo_train_path, index=False)
    sub_val.to_csv(demo_val_path, index=False)
    sub_test.to_csv(demo_test_path, index=False)

    print(
        f"   Created subsets: Train={len(sub_train)}, Val={len(sub_val)}, Test={len(sub_test)}"
    )

    # Monkey-patch Config to use demo paths and settings
    Config.TRAIN_CSV = demo_train_path
    Config.VAL_CSV = demo_val_path
    Config.TEST_CSV = demo_test_path
    Config.CACHE_DIR = DEMO_CACHE_DIR
    Config.MODEL_PATH = os.path.join(DEMO_MODEL_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    # Reduce compute load
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2

    print("\n[3] Testing Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    batch = next(iter(train_loader))
    print("   Batch Keys:", list(batch.keys()))

    # Verify Shapes
    img_ax = batch["image_axial"]
    img_cor = batch["image_coronal"]
    tab = batch["tabular"]

    print(f"   Axial Image Shape: {img_ax.shape}")
    print(f"   Tabular Shape: {tab.shape}")

    assert img_ax.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect axial image shape"
    assert img_cor.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect coronal image shape"
    assert tab.ndim == 2, "Tabular data should be 2D"
    print("   Data Loading verification passed.")

    print("\n[4] Testing Model Forward Pass...")
    device = Config.DEVICE
    tab_dim = tab.shape[1]

    model = SLHDAN(tabular_input_dim=tab_dim).to(device)

    # Move batch to device
    img_ax = img_ax.to(device)
    img_cor = img_cor.to(device)
    tab = tab.to(device)

    # Forward
    alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

    print(
        f"   Output Shapes -> Alpha: {alpha.shape}, Sigma_Base: {sigma_base.shape}, Sigma_Growth: {sigma_growth.shape}"
    )

    # Assertions
    assert alpha.shape == (Config.BATCH_SIZE,), "Alpha shape mismatch"
    assert sigma_base.shape == (Config.BATCH_SIZE,), "Sigma Base shape mismatch"
    assert torch.all(sigma_base > 0), "Sigma Base must be positive (Softplus)"
    assert torch.all(sigma_growth > 0), "Sigma Growth must be positive (Softplus)"
    print("   Model architecture verification passed.")

    print("\n[5] Running Training Loop (Debug Mode)...")
    # We use debug=True to run only a few batches
    best_score = train_model(epochs=1, batch_size=Config.BATCH_SIZE, debug=True)

    print(f"   Training finished. Best Validation Score: {best_score}")
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print("   Checkpoint verified.")

    print("\n[6] Running Inference and Submission...")
    predict_and_submit(batch_size=Config.BATCH_SIZE)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission Rows: {len(sub_df)}")
    print("   Head:")
    print(sub_df.head(3))

    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"

    print("   Submission verification passed.")

    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
