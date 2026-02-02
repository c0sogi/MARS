import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.data_utils import DataUtils
from library.dataset import LungDataset
from library.model import TQSAN
from library.loss import LaplaceLogLikelihoodLoss
from library.train_eval import Trainer


def main():
    print("=== TQ-SAN Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    print("\n[1] Setting up configuration...")
    Config.setup()

    # Override defaults for a fast demonstration
    Config.epochs = 2
    Config.batch_size = 4
    Config.num_workers = 2

    # Use a temporary directory for this demo to avoid overwriting main work
    demo_dir = os.path.join(Config.working_dir, "demo_execution")
    Config.idea_dir = demo_dir
    Config.cache_dir = os.path.join(demo_dir, "cache")
    Config.model_save_path = os.path.join(demo_dir, "demo_model.pth")

    # Disable pretrained weights to prevent connection errors in offline environments
    Config.pretrained = False

    os.makedirs(Config.cache_dir, exist_ok=True)
    print(f"    Working directory: {demo_dir}")
    print(f"    Device: {Config.device}")

    # ---------------------------------------------------------
    # 2. Data Preparation (DataUtils)
    # ---------------------------------------------------------
    print("\n[2] Preparing Data Subset...")

    # Load full training metadata
    try:
        train_df = pd.read_csv(Config.train_csv)
    except FileNotFoundError:
        print(f"CRITICAL: Metadata file not found at {Config.train_csv}")
        return

    # Subset: Select 3 unique patients to keep processing time very short
    unique_patients = train_df["Patient"].unique()[:3]
    demo_df = train_df[train_df["Patient"].isin(unique_patients)].copy()
    print(
        f"    Selected {len(unique_patients)} patients ({len(demo_df)} samples) for demonstration."
    )

    # Run Data Preparation Pipeline
    # This caches images as .npy and processes tabular features
    data_dict = DataUtils.prepare_dataset(
        demo_df, Config.cache_dir, mode="train", load_cached_data=False
    )

    # Verification
    assert "meta" in data_dict, "Data dictionary missing 'meta' key"
    assert "img_paths" in data_dict, "Data dictionary missing 'img_paths' key"
    assert len(data_dict["img_paths"]) == len(
        demo_df
    ), "Mismatch in image paths and dataframe length"
    assert (
        data_dict["meta"].shape[1] == 4
    ), "Tabular features should have 4 columns (Age, Sex, Smoke, Percent)"
    print("    DataUtils verification passed.")

    # ---------------------------------------------------------
    # 3. Dataset Instantiation (LungDataset)
    # ---------------------------------------------------------
    print("\n[3] Initializing LungDataset...")

    dataset = LungDataset(data_dict, mode="train")

    # Verification: Check item structure
    sample = dataset[0]
    # Expected shape: (Channels, Height, Width) -> (3, 224, 224)
    assert sample["axial"].shape == (
        3,
        224,
        224,
    ), f"Incorrect Axial shape: {sample['axial'].shape}"
    assert sample["coronal"].shape == (
        3,
        224,
        224,
    ), f"Incorrect Coronal shape: {sample['coronal'].shape}"
    assert sample["target"].shape == (1,), "Target should be a scalar tensor"
    print("    Dataset verification passed. Sample shapes correct.")

    # ---------------------------------------------------------
    # 4. Model Initialization (TQSAN)
    # ---------------------------------------------------------
    print("\n[4] Initializing TQSAN Model...")

    model = TQSAN()
    model.to(Config.device)

    # Verification: Forward Pass
    # Create a small DataLoader to simulate a batch
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    axial = batch["axial"].to(Config.device)
    coronal = batch["coronal"].to(Config.device)
    meta = batch["meta"].to(Config.device)

    with torch.no_grad():
        preds = model(axial, coronal, meta)

    # Expected Output: [Batch, 3] -> [Alpha, Sigma_Base, Sigma_Growth]
    assert preds.shape == (
        2,
        3,
    ), f"Model output shape mismatch. Expected (2, 3), got {preds.shape}"
    print("    Model forward pass successful.")

    # ---------------------------------------------------------
    # 5. Loss Function (LaplaceLogLikelihoodLoss)
    # ---------------------------------------------------------
    print("\n[5] Verifying Loss Function...")

    criterion = LaplaceLogLikelihoodLoss()

    target = batch["target"].to(Config.device)
    dt = batch["dt"].to(Config.device)
    base_fvc = batch["base_fvc"].to(Config.device)

    loss = criterion(preds, target, dt, base_fvc)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.dim() == 0, "Loss should be a scalar"
    print(f"    Loss calculation successful. Value: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 6. Training Loop (Trainer)
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop...")

    # For demo, we use the same small dataset for train and val
    train_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    trainer = Trainer(model, train_loader, val_loader, device=Config.device)

    # Run fit (Config.epochs is set to 2)
    trainer.fit(epochs=Config.epochs)

    assert os.path.exists(Config.model_save_path), "Model checkpoint was not saved."
    print("    Training loop completed successfully.")

    # ---------------------------------------------------------
    # 7. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[7] Running Inference Demo...")

    # Load test metadata
    test_df = pd.read_csv(Config.test_csv)

    # Subset test data (2 patients)
    test_patients = test_df["Patient"].unique()[:2]
    demo_test_df = test_df[test_df["Patient"].isin(test_patients)].copy()
    print(f"    Predicting for {len(demo_test_df)} rows (Test Subset).")

    # Prepare test data (Mode='test' ensures correct tabular processing using saved scalers)
    test_data_dict = DataUtils.prepare_dataset(
        demo_test_df, Config.cache_dir, mode="test", load_cached_data=False
    )

    test_dataset = LungDataset(test_data_dict, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # Load the best model saved during training
    model.load_state_dict(
        torch.load(Config.model_save_path, map_location=Config.device)
    )
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            axial = batch["axial"].to(Config.device)
            coronal = batch["coronal"].to(Config.device)
            meta = batch["meta"].to(Config.device)

            # Metadata for reconstruction
            dt = batch["dt"].to(Config.device)
            base_fvc = batch["base_fvc"].to(Config.device)

            # Forward pass
            out = model(axial, coronal, meta)

            # Decode outputs
            alpha = out[:, 0:1]
            sigma_base = out[:, 1:2]
            sigma_growth = out[:, 2:3]

            # FVC = Baseline + Alpha * Delta_Time
            fvc_pred = base_fvc + alpha * dt

            # Sigma = Sigma_Base + Sigma_Growth * |Delta_Time|
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            # Collect results
            batch_res = torch.cat([fvc_pred, sigma_pred], dim=1).cpu().numpy()
            predictions.append(batch_res)

    final_preds = np.concatenate(predictions, axis=0)

    # Verification
    assert final_preds.shape == (len(demo_test_df), 2), "Prediction shape mismatch"
    print("    Inference successful.")
    print(f"    Sample Prediction (FVC, Conf): {final_preds[0]}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
