import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders, OSICDataset
from library.model import RIDSNet
from library.train import run_training


def main():
    print("=== Starting OSIC Pulmonary Fibrosis Progression Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo execution...")

    # Override Config parameters for speed and isolation
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8  # Small batch size for demonstration
    Config.NUM_WORKERS = 2

    # Redirect outputs to a demo-specific directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create necessary directories
    Config.setup_directories()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[2] Loading data and verifying pipeline...")

    # Load dataloaders
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Verify Train Batch
    batch = next(iter(train_loader))
    images = batch["image"]
    tabular = batch["tabular"]
    targets = batch["target"]

    print(f"    Batch Image Shape: {images.shape}")  # Expected: (B, 3, 260, 260)
    print(f"    Batch Tabular Shape: {tabular.shape}")  # Expected: (B, 7)
    print(f"    Batch Target Shape: {targets.shape}")  # Expected: (B,)

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Image shape mismatch"
    assert tabular.shape == (Config.BATCH_SIZE, 7), "Tabular feature shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE,), "Target shape mismatch"

    print("    Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n[3] Initializing model and verifying forward pass...")

    model = RIDSNet().to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    tabular = tabular.to(Config.DEVICE)

    # Forward pass
    preds = model(images, tabular)

    print(f"    Prediction Shape: {preds.shape}")  # Expected: (B, 2)

    # Assertions
    assert preds.shape == (Config.BATCH_SIZE, 2), "Prediction shape mismatch"
    assert not torch.isnan(preds).any(), "Model produced NaNs"

    print("    Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Loss & Metric Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Loss and Metric functions...")

    # Loss Verification
    criterion = LaplaceLogLikelihoodLoss()
    loss = criterion(preds, targets.to(Config.DEVICE))
    print(f"    Calculated Loss (Random Init): {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # Metric Verification (Manual Check)
    # Scenario: True=2000, Pred=2000 (Delta=0), Sigma=50 (Clipped to 70)
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    y_true_dummy = np.array([2000])
    y_pred_dummy = np.array([2000])
    sigma_dummy = np.array([50])

    metric_val = calculate_metric(y_true_dummy, y_pred_dummy, sigma_dummy)
    print(f"    Calculated Metric (Perfect Pred, Low Conf): {metric_val:.4f}")

    # Expected value check
    expected_val = -np.log(np.sqrt(2) * 70)
    assert np.isclose(
        metric_val, expected_val, atol=1e-3
    ), "Metric calculation incorrect"

    print("    Loss and Metric verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (2 Epochs)...")

    # Run the training routine provided in library
    run_training()

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("    Training complete and model saved.")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Load the best model
    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    # Prepare Test Data
    # The task requires predicting for every possible week.
    # We load the test metadata (which contains only baselines) and expand it.
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Generate weeks range (e.g., -12 to 133)
    # For demo speed, we'll use a smaller range, e.g., -12 to 12
    # In full submission, this would be range(-12, 134)
    weeks_to_predict = list(range(-12, 14))

    print(
        f"    Expanding {len(test_df)} test patients over {len(weeks_to_predict)} weeks..."
    )

    expanded_rows = []
    for _, row in test_df.iterrows():
        patient = row["Patient"]
        # We copy the baseline info for every week
        # The OSICDataset logic handles calculating 'Weeks' relative to baseline
        for w in weeks_to_predict:
            new_row = row.copy()
            new_row["Weeks"] = w
            expanded_rows.append(new_row)

    expanded_test_df = pd.DataFrame(expanded_rows)

    # Create Dataset for inference
    inference_ds = OSICDataset(expanded_test_df, stats, mode="test")
    inference_loader = torch.utils.data.DataLoader(
        inference_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Prediction Loop
    results = []
    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    with torch.no_grad():
        for batch in inference_loader:
            imgs = batch["image"].to(Config.DEVICE)
            tabs = batch["tabular"].to(Config.DEVICE)
            p_ids = batch["patient_id"]

            # Predict
            out = model(imgs, tabs)
            mu_norm = out[:, 0]
            raw_sigma = out[:, 1]

            # Inverse Transform
            # Sigma: Softplus -> Scale
            sigma_est = torch.nn.functional.softplus(raw_sigma) * fvc_std
            # Mu: Unnormalize
            fvc_est = mu_norm * fvc_std + fvc_mean

            # Collect results
            # We need to reconstruct the 'Weeks' from the dataset logic or track it.
            # Since the loader order matches the dataframe order, we can use the dataframe index.
            # However, batch processing makes direct index matching tricky without an index tracker.
            # A simpler way for this demo is to rely on the fact that we iterate sequentially.
            pass  # We will zip results below

            # Convert to numpy
            fvc_np = fvc_est.cpu().numpy()
            sigma_np = sigma_est.cpu().numpy()

            for i in range(len(p_ids)):
                results.append((fvc_np[i], sigma_np[i]))

    # Attach predictions back to the dataframe
    expanded_test_df["FVC_Pred"] = [x[0] for x in results]
    expanded_test_df["Confidence"] = [x[1] for x in results]

    # Format for submission
    submission = pd.DataFrame()
    submission["Patient_Week"] = (
        expanded_test_df["Patient"] + "_" + expanded_test_df["Weeks"].astype(str)
    )
    submission["FVC"] = expanded_test_df["FVC_Pred"]
    submission["Confidence"] = expanded_test_df["Confidence"]

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Submission Shape: {submission.shape}")
    print("    Head:")
    print(submission.head())

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
