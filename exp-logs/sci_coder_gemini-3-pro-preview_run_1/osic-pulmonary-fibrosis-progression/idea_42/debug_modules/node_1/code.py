import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model_lib
import library.train as train_lib


def main():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("\n[1] Overriding Configuration for Fast Demonstration...")
    # Monkey-patching config to run a minimal version
    config.DEBUG = True
    config.DEBUG_DATASET_SIZE = 10  # Use only 10 samples
    config.BATCH_SIZE = 2  # Small batch size
    config.EPOCHS = 1  # Only 1 epoch
    config.PATIENCE = 1  # Minimal patience
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo

    print(f"Debug Mode: {config.DEBUG}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")

    # ==========================================
    # 2. Utility Verification
    # ==========================================
    print("\n[2] Verifying Utilities...")
    utils.seed_everything(42)

    # Test AverageMeter
    meter = utils.AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=2)
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    print("AverageMeter logic verified.")

    # Test Laplace Log Likelihood Loss
    # Formula: - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    # We minimize negative metric -> Loss = (sqrt(2) * delta / sigma) + ln(sqrt(2) * sigma)
    true_fvc = torch.tensor([2000.0])
    pred_fvc = torch.tensor([2100.0])  # Delta = 100
    pred_sigma = torch.tensor([70.0])  # Clipped at 70

    loss = utils.laplace_log_likelihood_loss(true_fvc, pred_fvc, pred_sigma)

    # Manual calc
    delta = 100.0
    sigma = 70.0
    sqrt_2 = np.sqrt(2)
    expected_loss = (sqrt_2 * delta / sigma) + np.log(sqrt_2 * sigma)

    # Allow small float tolerance
    assert torch.isclose(
        loss, torch.tensor(expected_loss, dtype=torch.float32), atol=1e-4
    ), f"Loss calculation mismatch. Got {loss.item()}, expected {expected_loss}"
    print(f"Loss function verified. Value: {loss.item():.4f}")

    # ==========================================
    # 3. Data Loading Demonstration
    # ==========================================
    print("\n[3] Demonstrating Data Loading...")
    train_loader, val_loader, test_loader = data.get_dataloaders()

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))
    axial, coronal, tab_vec, weeks_diff, base_fvc, target_fvc = batch

    # Verify Shapes
    # Images: (B, 3, 224, 224)
    assert axial.shape == (
        config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Axial shape mismatch: {axial.shape}"
    assert coronal.shape == (
        config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Coronal shape mismatch: {coronal.shape}"
    # Tabular: (B, 4)
    assert tab_vec.shape == (
        config.BATCH_SIZE,
        4,
    ), f"Tabular shape mismatch: {tab_vec.shape}"
    # Scalars: (B,)
    assert weeks_diff.shape == (config.BATCH_SIZE,), "Weeks diff shape mismatch"
    assert target_fvc.shape == (config.BATCH_SIZE,), "Target FVC shape mismatch"

    print("Data batch shapes verified successfully.")

    # ==========================================
    # 4. Model Demonstration
    # ==========================================
    print("\n[4] Demonstrating Model Forward Pass...")
    device = torch.device("cpu")  # Use CPU for simple demo
    model = model_lib.CRHDAN().to(device)
    model.eval()

    with torch.no_grad():
        # Move batch to device
        p_fvc, p_sigma = model(
            axial.to(device),
            coronal.to(device),
            tab_vec.to(device),
            weeks_diff.to(device),
            base_fvc.to(device),
        )

    # Check outputs
    assert p_fvc.shape == (
        config.BATCH_SIZE,
    ), f"Pred FVC shape mismatch: {p_fvc.shape}"
    assert p_sigma.shape == (
        config.BATCH_SIZE,
    ), f"Pred Sigma shape mismatch: {p_sigma.shape}"
    assert torch.all(torch.isfinite(p_fvc)), "Model produced NaN/Inf in FVC"
    assert torch.all(p_sigma > 0), "Sigma must be positive (Softplus)"

    print("Model forward pass successful.")
    print(f"Sample Predictions: FVC={p_fvc[0]:.2f}, Sigma={p_sigma[0]:.2f}")

    # ==========================================
    # 5. Training Step Demonstration
    # ==========================================
    print("\n[5] Demonstrating Training Step...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch manually using the library function
    # Note: train_one_epoch expects the loader, model, optimizer, device
    avg_loss = train_lib.train_one_epoch(train_loader, model, optimizer, device)
    print(f"Single epoch training completed. Avg Loss: {avg_loss:.4f}")

    # Run evaluation manually
    val_loss = train_lib.evaluate(val_loader, model, device)
    print(f"Evaluation completed. Val Loss: {val_loss:.4f}")

    # ==========================================
    # 6. Full Pipeline Execution
    # ==========================================
    print("\n[6] Running Full Training Pipeline (Mock)...")
    # This calls the main training loop in library.train
    # It will use the patched config (1 epoch)
    best_model_path = train_lib.train_model()

    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"Full pipeline finished. Model saved at: {best_model_path}")

    # ==========================================
    # 7. Inference & Submission
    # ==========================================
    print("\n[7] Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            # Test loader yields: (axial, coronal, tab_vec, weeks_diff, base_fvc, pat_week_id)
            ax = batch[0].to(device)
            cor = batch[1].to(device)
            tab = batch[2].to(device)
            wd = batch[3].to(device)
            bfvc = batch[4].to(device)
            pat_week_ids = batch[5]

            pred_fvc, pred_sigma = model(ax, cor, tab, wd, bfvc)

            # Collect results
            for i in range(len(pat_week_ids)):
                submission_rows.append(
                    {
                        "Patient_Week": pat_week_ids[i],
                        "FVC": pred_fvc[i].item(),
                        "Confidence": pred_sigma[i].item(),
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(submission_rows)

    # Verify columns
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in required_cols
    ), "Submission columns missing."

    # Save to working directory
    output_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    sub_df.to_csv(output_path, index=False)

    print(f"Submission generated with {len(sub_df)} rows.")
    print(f"Saved to: {output_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
