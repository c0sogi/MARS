import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import library modules
import library.config
import library.data
import library.model
import library.train
import library.utils


# ==========================================
# 1. Setup & Configuration Overrides
# ==========================================
def setup_demo_environment():
    """
    Creates a mini-dataset and overrides library configurations
    to ensure the demo runs quickly and uses temporary paths.
    """
    print("Setting up demo environment...")

    # Define paths
    demo_dir = "./working/demo"
    os.makedirs(demo_dir, exist_ok=True)

    mini_train_path = os.path.join(demo_dir, "train.csv")
    mini_val_path = os.path.join(demo_dir, "val.csv")
    mini_test_path = os.path.join(demo_dir, "test.csv")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_checkpoint_dir, exist_ok=True)

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample data (2 patients for train, 1 for val, 1 for test)
    # This ensures dataloaders are not empty but processing is fast
    train_patients = orig_train["Patient"].unique()[:2]
    val_patients = orig_val["Patient"].unique()[:1]
    test_patients = orig_test["Patient"].unique()[:1]

    mini_train = orig_train[orig_train["Patient"].isin(train_patients)].copy()
    mini_val = orig_val[orig_val["Patient"].isin(val_patients)].copy()
    mini_test = orig_test[orig_test["Patient"].isin(test_patients)].copy()

    # Save mini datasets
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(
        f"Mini-datasets created: Train={len(mini_train)}, Val={len(mini_val)}, Test={len(mini_test)}"
    )

    # --- Monkey-Patching Library Constants ---
    # We override the constants in the imported modules to point to our mini data

    # Patch library.data
    library.data.TRAIN_CSV = mini_train_path
    library.data.VAL_CSV = mini_val_path
    library.data.TEST_CSV = mini_test_path
    library.data.CACHE_DIR = demo_cache_dir

    # Patch library.model
    # Disable pretrained weights to avoid download/network issues during demo
    library.model.PRETRAINED = False

    # Patch library.config (used by other modules)
    library.config.CACHE_DIR = demo_cache_dir
    library.config.CHECKPOINT_DIR = demo_checkpoint_dir
    library.config.BEST_MODEL_PATH = os.path.join(demo_checkpoint_dir, "best_model.pth")
    library.config.BATCH_SIZE = 4  # Small batch size for demo

    return demo_checkpoint_dir


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Ensure reproducibility
    library.utils.seed_everything(42)

    # Setup data and paths
    ckpt_dir = setup_demo_environment()
    device = library.config.DEVICE
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("\n--- 2. Data Loading ---")
    # get_dataloaders handles caching (preprocessing images) and scaling
    train_loader, val_loader, test_loader, scaler = library.data.get_dataloaders(
        load_cached_data=True
    )

    print("DataLoaders initialized.")

    # Verify Train Loader
    imgs, tab, targets = next(iter(train_loader))
    print(
        f"Batch Shapes -> Images: {imgs.shape}, Tabular: {tab.shape}, Targets: {targets.shape}"
    )

    # Assertions to verify data integrity
    # Images: (Batch, N_Slices, H, W) -> (B, 3, 256, 256)
    assert (
        imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[2] == 256
    ), "Incorrect image tensor shape"
    # Tabular: (Batch, 5) -> [Baseline, Weeks, Age, Sex, Smoking]
    assert tab.ndim == 2 and tab.shape[1] == 5, "Incorrect tabular tensor shape"
    # Targets: (Batch,)
    assert targets.ndim == 1, "Incorrect target tensor shape"

    # ---------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # ---------------------------------------------------------
    print("\n--- 3. Model Instantiation & Forward Pass ---")
    model = library.model.SAPNet().to(device)
    print("SAPNet instantiated.")

    # Move batch to device
    imgs = imgs.to(device)
    tab = tab.to(device)

    # Forward pass
    mu, sigma = model(imgs, tab)

    print(f"Output Shapes -> Mu: {mu.shape}, Sigma: {sigma.shape}")

    # Assertions
    assert mu.shape == targets.shape, "Prediction shape mismatch"
    assert sigma.shape == targets.shape, "Uncertainty shape mismatch"
    # Sigma should be positive (softplus + floor used in model)
    assert torch.all(sigma > 0), "Sigma contains non-positive values"

    # ---------------------------------------------------------
    # 4. Loss Function Demonstration
    # ---------------------------------------------------------
    print("\n--- 4. Loss Calculation ---")
    criterion = library.train.LaplaceNLLLoss()
    targets = targets.to(device)

    loss = criterion(mu, sigma, targets)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.ndim == 0, "Loss should be a scalar"

    # ---------------------------------------------------------
    # 5. Metric Demonstration
    # ---------------------------------------------------------
    print("\n--- 5. Metric Calculation ---")
    # Simulate some values
    y_true_demo = np.array([2000, 2500, 3000])
    y_pred_demo = np.array([2100, 2400, 2900])  # Errors: 100, 100, 100
    sigma_demo = np.array([100, 50, 200])  # 50 will be clipped to 70

    score = library.utils.metric_laplace_log_likelihood(
        y_true_demo, y_pred_demo, sigma_demo
    )
    print(f"Metric Score (Manual Test): {score:.4f}")

    # Validation:
    # Case 2 (sigma=50 -> 70): Error=100. Metric term 1 = -sqrt(2)*100/70 = -2.02. Term 2 = -ln(sqrt(2)*70) = -4.59. Sum ~ -6.61
    # Case 1 (sigma=100): Error=100. Metric term 1 = -1.414. Term 2 = -4.95. Sum ~ -6.36
    assert score < 0, "Metric should be negative"

    # ---------------------------------------------------------
    # 6. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n--- 6. Training Loop (1 Epoch) ---")
    # Setup optimizer as in library.train.train_model
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch using the library function
    train_loss = library.train.train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Epoch Complete. Train Loss: {train_loss:.4f}")

    # ---------------------------------------------------------
    # 7. Evaluation
    # ---------------------------------------------------------
    print("\n--- 7. Evaluation ---")
    val_score = library.train.evaluate(model, val_loader, scaler, device)
    print(f"Validation Score: {val_score:.4f}")

    # Save the model (required for submission generation step)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_means": scaler.means,
            "scaler_stds": scaler.stds,
            "best_score": val_score,
        },
        library.config.BEST_MODEL_PATH,
    )
    print(f"Model saved to {library.config.BEST_MODEL_PATH}")

    # ---------------------------------------------------------
    # 8. Inference / Submission Generation
    # ---------------------------------------------------------
    print("\n--- 8. Inference (Submission Generation) ---")

    # We will simulate the logic inside generate_submission using the test_loader
    model.eval()
    results = []

    print("Predicting on test set...")
    with torch.no_grad():
        for imgs, tab, patient_week_ids in test_loader:
            imgs = imgs.to(device)
            tab = tab.to(device)

            # Predict (Scaled values)
            mu_scaled, sigma_scaled = model(imgs, tab)

            # Inverse Transform to get mL
            mu = scaler.inverse_transform_target(mu_scaled.cpu().numpy())
            sigma = scaler.inverse_transform_sigma(sigma_scaled.cpu().numpy())

            # Post-process sigma
            sigma = np.maximum(sigma, library.config.METRIC_CLIP_SIGMA)

            # Collect results
            for pw, fvc, conf in zip(patient_week_ids, mu, sigma):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Convert to DataFrame
    sub_df = pd.DataFrame(results)
    print(f"Generated predictions for {len(sub_df)} rows.")
    print("Sample predictions:")
    print(sub_df.head(3))

    # Assertions
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert len(sub_df) > 0, "Submission dataframe is empty"

    print("\nDemo completed successfully.")
