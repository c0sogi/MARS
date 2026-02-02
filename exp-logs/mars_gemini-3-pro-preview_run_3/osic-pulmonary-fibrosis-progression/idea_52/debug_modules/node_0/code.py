import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import AILRNet
from library.loss import LaplaceLogLikelihoodLoss
from library.engine import fit, predict


def run_demo():
    print("=== Starting AILR-Net Demo Execution ===")

    # 1. Setup Environment
    seed_everything(42)
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working directory: {demo_dir}")

    # 2. Create Data Subsets for Speed
    # We will sample a few rows from the existing metadata to create a 'mini' dataset.
    print("\n[Step 1] Creating mini-datasets for rapid demonstration...")

    # Load original metadata
    orig_train = pd.read_csv(Config.TRAIN_CSV)
    orig_val = pd.read_csv(Config.VAL_CSV)
    orig_test = pd.read_csv(Config.TEST_CSV)
    orig_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Subset: 16 rows for train, 8 for val, 2 for test
    # We select by Patient to ensure we don't break the patient grouping
    train_patients = orig_train["Patient"].unique()[:5]
    val_patients = orig_val["Patient"].unique()[:2]
    test_patients = orig_test["Patient"].unique()[:2]

    mini_train = orig_train[orig_train["Patient"].isin(train_patients)].copy()
    mini_val = orig_val[orig_val["Patient"].isin(val_patients)].copy()
    mini_test = orig_test[orig_test["Patient"].isin(test_patients)].copy()

    # Filter sample submission for the test patients
    mini_sub = orig_sub[
        orig_sub["Patient_Week"].apply(lambda x: x.split("_")[0] in test_patients)
    ].copy()

    # Save mini datasets
    mini_train_path = os.path.join(demo_dir, "train.csv")
    mini_val_path = os.path.join(demo_dir, "val.csv")
    mini_test_path = os.path.join(demo_dir, "test.csv")
    mini_sub_path = os.path.join(demo_dir, "sample_submission.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)
    mini_sub.to_csv(mini_sub_path, index=False)

    print(f"  Train samples: {len(mini_train)}")
    print(f"  Val samples: {len(mini_val)}")
    print(f"  Test samples: {len(mini_test)}")

    # 3. Override Config
    print("\n[Step 2] Overriding Configuration for Demo...")
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path
    Config.SAMPLE_SUBMISSION = mini_sub_path

    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 4. Initialize DataLoaders
    print("\n[Step 3] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")

    # 5. Verify Model and Forward Pass
    print("\n[Step 4] Verifying Model Architecture and Forward Pass...")
    device = Config.DEVICE
    model = AILRNet().to(device)

    # Fetch one batch
    batch = next(iter(train_loader))
    images, restricted, context, targets = [x.to(device) for x in batch]

    print(f"  Input Shapes:")
    print(f"    Images: {images.shape}")
    print(f"    Restricted: {restricted.shape}")
    print(f"    Context: {context.shape}")

    # Forward
    mu, sigma = model(images, restricted, context)

    print(f"  Output Shapes:")
    print(f"    Mu: {mu.shape}")
    print(f"    Sigma: {sigma.shape}")

    assert mu.shape == targets.shape, "Mu shape mismatch"
    assert sigma.shape == targets.shape, "Sigma shape mismatch"
    assert not torch.isnan(mu).any(), "Model produced NaN in Mu"

    # 6. Verify Loss
    print("\n[Step 5] Verifying Loss Function...")
    loss_fn = LaplaceLogLikelihoodLoss()
    loss = loss_fn(mu, sigma, targets)
    print(f"  Initial Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 7. Run Training Loop
    print("\n[Step 6] Running Training Loop (Fit)...")
    fit(model, train_loader, val_loader, device, epochs=Config.EPOCHS)

    # Check if checkpoint exists
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print("  Checkpoint saved successfully.")
    else:
        raise FileNotFoundError("Best model checkpoint was not created.")

    # 8. Run Inference
    print("\n[Step 7] Running Inference (Predict)...")
    # Reload best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    predict(model, test_loader, device)

    submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_file):
        sub_df = pd.read_csv(submission_file)
        print(f"  Submission generated with {len(sub_df)} rows.")
        print(f"  Columns: {list(sub_df.columns)}")

        # Validate format
        assert "Patient_Week" in sub_df.columns
        assert "FVC" in sub_df.columns
        assert "Confidence" in sub_df.columns
        assert not sub_df.isnull().values.any(), "Submission contains null values"
    else:
        raise FileNotFoundError("Submission file was not created.")

    # 9. Verify Metric Calculation Logic
    print("\n[Step 8] Verifying Metric Calculation Logic...")
    # Test case:
    # True: 2000
    # Pred: 2100 (Delta = 100)
    # Sigma: 50 (Clipped to 70)
    # Formula: - (sqrt(2) * 100) / 70 - ln(sqrt(2) * 70)
    #        = - (1.41421 * 100) / 70 - ln(1.41421 * 70)
    #        = - 2.0203 - ln(98.9949)
    #        = - 2.0203 - 4.5950
    #        = - 6.6153

    fvc_true = np.array([2000.0])
    fvc_pred = np.array([2100.0])
    sigma_pred = np.array([50.0])  # Should clip to 70

    calc_score = calculate_metric(fvc_true, fvc_pred, sigma_pred)

    # Manual calc
    sigma_clipped = 70.0
    delta = 100.0
    expected_score = -(np.sqrt(2) * delta) / sigma_clipped - np.log(
        np.sqrt(2) * sigma_clipped
    )

    print(f"  Calculated Score: {calc_score:.4f}")
    print(f"  Expected Score:   {expected_score:.4f}")

    np.testing.assert_almost_equal(
        calc_score, expected_score, decimal=4, err_msg="Metric calculation mismatch"
    )
    print("  Metric logic verified.")

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
