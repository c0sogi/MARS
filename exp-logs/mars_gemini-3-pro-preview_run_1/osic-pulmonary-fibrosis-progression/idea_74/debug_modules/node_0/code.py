import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import PCCGNet
from library.engine import train_one_epoch, evaluate, predict


def main():
    print("=== Starting Library Verification Script ===")

    # ---------------------------------------------------------
    # 1. Configuration for Fast Execution
    # ---------------------------------------------------------
    # Override defaults to ensure the script runs in < 1 minute
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 4  # Only use 4 patients for training
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small test

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[1/5] Verifying Data Loading...")
    # debug=True loads a tiny subset of data defined by DEBUG_SAMPLE_SIZE
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch a single batch to inspect
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError(
            "Train loader is empty! Check dataset paths and debug split logic."
        )

    # Verify keys
    required_keys = ["img_ax", "img_cor", "tabular", "target", "weeks", "base_fvc"]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify shapes
    # Image: (Batch, 3, 224, 224)
    assert batch["img_ax"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image Shape: {batch['img_ax'].shape}"
    # Tabular: (Batch, 6) -> Age, Sex, Percent, Smoke_Ex, Smoke_Never, Smoke_Current
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Incorrect Tabular Shape: {batch['tabular'].shape}"

    print("   -> Data Loader output verified successfully.")

    # ---------------------------------------------------------
    # 3. Model & Loss Verification
    # ---------------------------------------------------------
    print("\n[2/5] Verifying Model and Loss...")
    model = PCCGNet().to(device)
    loss_fn = LaplaceLogLikelihoodLoss().to(device)

    # Move batch to device
    img_ax = batch["img_ax"].to(device)
    img_cor = batch["img_cor"].to(device)
    tabular = batch["tabular"].to(device)
    weeks = batch["weeks"].to(device)
    base_fvc = batch["base_fvc"].to(device)
    targets = batch["target"].to(device)

    # Forward Pass
    preds = model(img_ax, img_cor, tabular, weeks, base_fvc)

    # Check Output Shape: (Batch, 2) -> [FVC, Confidence]
    assert preds.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Incorrect Model Output Shape: {preds.shape}"

    # Compute Loss
    loss = loss_fn(preds, targets)
    assert not torch.isnan(loss).any(), "Loss contains NaNs"
    assert loss.ndim == 0, "Loss should be a scalar"

    print(f"   -> Forward pass successful. Initial Loss: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 4. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[3/5] Verifying Training Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch of training
    train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
    print(f"   -> Training step complete. Avg Train Loss: {train_loss:.4f}")

    # Run evaluation
    val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)
    print(f"   -> Evaluation complete. Val Metric: {val_metric:.4f}")

    # ---------------------------------------------------------
    # 5. Inference & Submission Verification
    # ---------------------------------------------------------
    print("\n[4/5] Verifying Inference pipeline...")
    # Predict on the debug test set
    fvc_preds, conf_preds = predict(model, test_loader, device)

    assert len(fvc_preds) == len(conf_preds), "Prediction arrays length mismatch"
    assert not np.isnan(fvc_preds).any(), "Predictions contain NaNs"

    print("\n[5/5] Generating Demo Submission...")
    # Load the test metadata to align predictions
    test_df = pd.read_csv(Config.TEST_CSV)

    # In debug mode, get_dataloaders filters the dataframe to the first 5 patients.
    # We must filter our reference dataframe similarly to match the predictions.
    test_patients = test_df["Patient"].unique()[:5]
    test_subset = test_df[test_df["Patient"].isin(test_patients)].reset_index(drop=True)

    # Verify lengths match
    if len(test_subset) != len(fvc_preds):
        print(
            f"Warning: Prediction count ({len(fvc_preds)}) differs from subset count ({len(test_subset)})."
        )
        # Truncate to match for demonstration if loader dropped last (though test loader shouldn't)
        min_len = min(len(test_subset), len(fvc_preds))
        test_subset = test_subset.iloc[:min_len]
        fvc_preds = fvc_preds[:min_len]
        conf_preds = conf_preds[:min_len]

    # Construct submission dataframe
    submission = pd.DataFrame(
        {
            "Patient_Week": test_subset["Patient_Week"],
            "FVC": fvc_preds,
            "Confidence": conf_preds,
        }
    )

    # Save
    out_file = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(out_file, index=False)

    print(f"   -> Submission saved to: {out_file}")
    print(f"   -> First 3 rows:\n{submission.head(3)}")

    print("\n=== Verification Complete: All components functioning correctly. ===")


if __name__ == "__main__":
    main()
