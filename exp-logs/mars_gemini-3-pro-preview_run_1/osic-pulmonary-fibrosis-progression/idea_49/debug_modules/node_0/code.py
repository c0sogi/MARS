import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders, OSICDataset
from library.model import SLHDAN
from library.train import loss_fn


def demo_main():
    print("=" * 40)
    print("SLHDAN Library Demonstration")
    print("=" * 40)

    # 1. Setup and Configuration Override
    print("\n[1] Setting up Configuration...")
    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = (
        0  # Use 0 for simple debugging to avoid multiprocessing overhead
    )

    # Ensure working directories exist (Config creates them on import, but good to be safe)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration configured for fast debugging.")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            debug=Config.DEBUG,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )
        print("DataLoaders initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize DataLoaders: {e}")
        raise e

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Contents
    required_keys = [
        "img_ax",
        "img_cor",
        "tabular",
        "target",
        "time_delta",
        "baseline_fvc",
        "patient_id",
        "week",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Shapes
    # Images: (B, 3, 224, 224) - 3 channels because Tri-Slab
    assert batch["img_ax"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image shape: {batch['img_ax'].shape}"
    assert batch["img_cor"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Coronal Image shape: {batch['img_cor'].shape}"

    # Tabular: (B, 7)
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        7,
    ), f"Incorrect Tabular shape: {batch['tabular'].shape}"

    print(f"Batch verification passed. Batch Size: {Config.BATCH_SIZE}")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Verifying Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = SLHDAN().to(device)

    # Move batch to device
    img_ax = batch["img_ax"].to(device)
    img_cor = batch["img_cor"].to(device)
    tabular = batch["tabular"].to(device)
    time_delta = batch["time_delta"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)
    target = batch["target"].to(device)

    # Forward Pass
    fvc_pred, sigma_pred = model(img_ax, img_cor, tabular, time_delta, baseline_fvc)

    # Verify Output Shapes
    assert fvc_pred.shape == (
        Config.BATCH_SIZE,
    ), f"Output FVC shape mismatch: {fvc_pred.shape}"
    assert sigma_pred.shape == (
        Config.BATCH_SIZE,
    ), f"Output Sigma shape mismatch: {sigma_pred.shape}"

    # Verify Values are Finite
    assert torch.all(torch.isfinite(fvc_pred)), "FVC predictions contain NaNs or Infs"
    assert torch.all(
        torch.isfinite(sigma_pred)
    ), "Sigma predictions contain NaNs or Infs"

    print("Model forward pass successful.")
    print(f"Predicted FVC (first sample): {fvc_pred[0].item():.2f}")
    print(f"Predicted Sigma (first sample): {sigma_pred[0].item():.2f}")

    # 4. Loss and Metric Calculation
    print("\n[4] Verifying Loss and Metric...")

    # Calculate Loss
    loss = loss_fn(fvc_pred, sigma_pred, target)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss is not finite"
    print(f"Calculated Loss: {loss.item():.4f}")

    # Calculate Metric (Score Function)
    # Convert to numpy/list as expected by utils.score_function
    y_true = target.cpu().detach().numpy()
    y_pred = fvc_pred.cpu().detach().numpy()
    sigma = sigma_pred.cpu().detach().numpy()

    metric = score_function(y_true, y_pred, sigma)
    assert isinstance(metric, float) or isinstance(
        metric, np.floating
    ), "Metric should be a float"
    print(f"Calculated Metric Score: {metric:.4f}")

    # 5. Optimization Step
    print("\n[5] Verifying Optimization Step...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    loss.backward()

    # Check if gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients computed after backward pass"

    optimizer.step()
    print("Optimizer step completed successfully.")

    # 6. Submission Simulation
    print("\n[6] Simulating Submission Generation...")
    model.eval()
    results = []

    # Use test loader (which uses metadata/test.csv)
    # Since we set DEBUG=True, this will be small
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            b_img_ax = batch["img_ax"].to(device)
            b_img_cor = batch["img_cor"].to(device)
            b_tabular = batch["tabular"].to(device)
            b_time_delta = batch["time_delta"].to(device)
            b_baseline_fvc = batch["baseline_fvc"].to(device)

            pids = batch["patient_id"]
            weeks = batch["week"]

            p_fvc, p_sigma = model(
                b_img_ax, b_img_cor, b_tabular, b_time_delta, b_baseline_fvc
            )

            p_fvc = p_fvc.cpu().numpy()
            p_sigma = p_sigma.cpu().numpy()

            for idx in range(len(pids)):
                results.append(
                    {
                        "Patient_Week": f"{pids[idx]}_{weeks[idx].item()}",
                        "FVC": p_fvc[idx],
                        "Confidence": p_sigma[idx],
                    }
                )

            # Just do one batch for demo
            break

    df_res = pd.DataFrame(results)
    print(f"Generated {len(df_res)} predictions.")
    print("Sample Prediction:")
    print(df_res.head(1).to_string(index=False))

    assert "Patient_Week" in df_res.columns
    assert "FVC" in df_res.columns
    assert "Confidence" in df_res.columns

    print("\n" + "=" * 40)
    print("DEMONSTRATION COMPLETE: All components verified.")
    print("=" * 40)


if __name__ == "__main__":
    demo_main()
