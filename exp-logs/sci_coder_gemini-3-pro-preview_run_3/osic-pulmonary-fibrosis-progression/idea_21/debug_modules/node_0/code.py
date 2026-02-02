import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Ensure the current directory is in the path for module imports
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, metric_laplace_log_likelihood
from library.data import get_dataloaders, get_test_loader
from library.model import RCRFNet
from library.train import LaplaceNLLLoss, train_one_epoch, validate


def run_demo():
    print("=== Starting RCRF-Net Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("[1] Setting up configuration for fast demo execution...")

    # Define demo directories
    DEMO_DIR = "./working/demo"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")

    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    # Override Config for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Setup directories based on new config
    Config.setup()

    # Reduce computational load
    Config.IMG_SIZE = 64  # Small images for speed
    Config.BATCH_SIZE = 2  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Main process only
    Config.DEBUG = True

    # Set seed
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Subsetting (Create mini-datasets)
    # -------------------------------------------------------------------------
    print("[2] Creating mini-datasets from metadata...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Select a few patients for training and validation
    # We use patients that definitely exist in the input/train directory
    train_patients = orig_train["Patient"].unique()[:3]
    val_patients = orig_val["Patient"].unique()[:2]
    test_patients = orig_test["Patient"].unique()[:2]

    sub_train = orig_train[orig_train["Patient"].isin(train_patients)].copy()
    sub_val = orig_val[orig_val["Patient"].isin(val_patients)].copy()
    sub_test = orig_test[orig_test["Patient"].isin(test_patients)].copy()

    # Save to demo location
    demo_train_path = os.path.join(DEMO_META_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_META_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_META_DIR, "test.csv")

    sub_train.to_csv(demo_train_path, index=False)
    sub_val.to_csv(demo_val_path, index=False)
    sub_test.to_csv(demo_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_CSV = demo_train_path
    Config.VAL_CSV = demo_val_path
    Config.TEST_CSV = demo_test_path

    print(f"    Train subset: {len(sub_train)} rows")
    print(f"    Val subset:   {len(sub_val)} rows")
    print(f"    Test subset:  {len(sub_test)} rows")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Verify Batch Structure
    print("    Verifying batch structure...")
    imgs, clinical, t_rel, targets = next(iter(train_loader))

    print(
        f"    Image shape: {imgs.shape} (Expected: [{Config.BATCH_SIZE}, 3, {Config.IMG_SIZE}, {Config.IMG_SIZE}])"
    )
    print(
        f"    Clinical shape: {clinical.shape} (Expected: [{Config.BATCH_SIZE}, {Config.CLINICAL_INPUT_DIM}])"
    )
    print(f"    Target shape: {targets.shape}")

    # Assertions
    assert imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Image Shape"
    assert clinical.shape == (
        Config.BATCH_SIZE,
        Config.CLINICAL_INPUT_DIM,
    ), "Incorrect Clinical Shape"
    assert t_rel.shape[0] == Config.BATCH_SIZE, "Incorrect Time Shape"

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Initializing RCRFNet model...")
    device = torch.device("cpu")  # Use CPU for simple demo stability/portability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("    Using CUDA.")

    model = RCRFNet().to(device)

    print("    Running forward pass...")
    imgs = imgs.to(device)
    clinical = clinical.to(device)
    t_rel = t_rel.to(device)

    mu, sigma = model(imgs, clinical, t_rel)

    print(f"    Output mu shape: {mu.shape}")
    print(f"    Output sigma shape: {sigma.shape}")

    # Assertions
    assert mu.shape == (Config.BATCH_SIZE, 1), "Incorrect Mu Output Shape"
    assert sigma.shape == (Config.BATCH_SIZE, 1), "Incorrect Sigma Output Shape"
    assert torch.all(sigma > 0), "Sigma must be positive (Softplus)"

    # -------------------------------------------------------------------------
    # 5. Loss & Metric
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Loss and Metric...")
    criterion = LaplaceNLLLoss()
    targets = targets.to(device).view(-1, 1)

    loss = criterion(mu, sigma, targets)
    print(f"    Computed Loss: {loss.item():.4f}")

    # Check metric function
    # Convert to numpy and scale back for metric calculation (just for demo logic)
    y_true_np = targets.detach().cpu().numpy().flatten()
    y_pred_np = mu.detach().cpu().numpy().flatten()
    sigma_np = sigma.detach().cpu().numpy().flatten()

    metric_val = metric_laplace_log_likelihood(y_true_np, y_pred_np, sigma_np)
    print(f"    Metric Score (Raw): {metric_val:.4f}")

    # -------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Epoch 1 Train Loss: {train_loss:.4f}")

    print("    Running Validation...")
    val_score = validate(model, val_loader, device)
    print(f"    Validation Score: {val_score:.4f}")

    # Save dummy checkpoint for inference step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # -------------------------------------------------------------------------
    # 7. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference on Test Set...")

    test_loader, test_df_expanded = get_test_loader(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    model.eval()
    predictions = []
    confidences = []

    with torch.no_grad():
        for imgs, clinical, t_rel, _ in test_loader:
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            t_rel = t_rel.to(device)

            mu, sigma = model(imgs, clinical, t_rel)

            # Inverse transform
            mu_real = mu.cpu().numpy() * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_real = sigma.cpu().numpy() * Config.TARGET_STD

            predictions.extend(mu_real.flatten())
            confidences.extend(sigma_real.flatten())

    # Create submission dataframe
    submission = pd.DataFrame(
        {
            "Patient_Week": test_df_expanded["Patient_Week"],
            "FVC": predictions,
            "Confidence": confidences,
        }
    )

    print(f"    Generated {len(submission)} predictions.")
    print("    Sample predictions:")
    print(submission.head(3))

    # Verify format
    assert "Patient_Week" in submission.columns
    assert "FVC" in submission.columns
    assert "Confidence" in submission.columns
    assert len(submission) == len(test_df_expanded)

    # Save
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"    Submission saved to {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
