import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import provided library components
from library.config import Config
from library.utils import seed_everything, score_function, count_parameters
from library.data import OSICDataset
from library.model import SCARNet
from library.loss import StandardizedLaplaceLoss
from library.train import Trainer


def demo_pipeline():
    print("--- Starting SCAR-Net Pipeline Demo ---")

    # 1. Setup and Config Overrides for Speed
    seed_everything(Config.SEED)

    # Override Config for a fast demo run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.BACKBONE_PRETRAINED = (
        False  # Avoid downloading weights if internet is restricted
    )

    print(f"Configuration set: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Data Pipeline Verification
    print("\n[1/5] Verifying Data Pipeline...")

    # Load a small subset of data to avoid processing all DICOMs
    full_train_df = pd.read_csv(Config.TRAIN_CSV)
    demo_df = full_train_df.head(4).copy()  # Select top 4 rows

    # Define simple transforms
    transforms = A.Compose(
        [A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE), ToTensorV2()]
    )

    # Instantiate Dataset
    dataset = OSICDataset(
        demo_df,
        mode="train",
        transform=transforms,
        cache_dir=Config.CACHE_DIR,
        load_cached=False,  # Force processing to test pipeline
    )

    # Instantiate Loader
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Fetch one batch
    images, tabular, targets = next(iter(loader))

    print(
        f"  Batch Shapes -> Images: {images.shape}, Tabular: {tabular.shape}, Targets: {targets.shape}"
    )

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Image tensor shape mismatch"
    assert tabular.shape == (Config.BATCH_SIZE, 5), "Tabular feature shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE, 1), "Target shape mismatch"
    print("  Data Pipeline Verified.")

    # 3. Model Architecture Verification
    print("\n[2/5] Verifying Model Architecture...")

    model = SCARNet()
    model.to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    tabular = tabular.to(Config.DEVICE)

    # Forward Pass
    preds = model(images, tabular)

    print(f"  Output Shape: {preds.shape}")
    print(f"  Parameter Count: {count_parameters(model):,}")

    # Assertions
    assert preds.shape == (Config.BATCH_SIZE, 2), "Model output shape must be (B, 2)"
    assert not torch.isnan(preds).any(), "Model output contains NaNs"

    # Check if sigma (2nd column) is positive (Softplus + Floor used in model)
    sigma_preds = preds[:, 1]
    assert (
        sigma_preds >= Config.SIGMA_FLOOR_STD
    ).all(), "Sigma predictions violate floor constraint"
    print("  Model Architecture Verified.")

    # 4. Loss Function Verification
    print("\n[3/5] Verifying Loss Function...")

    criterion = StandardizedLaplaceLoss()
    targets = targets.to(Config.DEVICE)

    loss = criterion(preds, targets)

    print(f"  Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() != 0, "Loss is zero (highly unlikely)"
    print("  Loss Function Verified.")

    # 5. Metric Verification
    print("\n[4/5] Verifying Metric Calculation...")

    # Create dummy data
    # Case 1: Perfect prediction
    y_true = np.array([2000, 3000])
    y_pred = np.array([2000, 3000])
    sigma = np.array([70, 70])  # Minimum sigma

    # Metric formula: - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
    # If delta=0, sigma=70: - ln(sqrt(2)*70) = - ln(98.99) approx -4.595
    score_perfect = score_function(y_true, y_pred, sigma)

    # Case 2: Large error (should be clipped at 1000)
    y_true_bad = np.array([2000])
    y_pred_bad = np.array([4000])  # Delta = 2000 -> clipped to 1000
    sigma_bad = np.array([100])

    score_bad = score_function(y_true_bad, y_pred_bad, sigma_bad)

    print(f"  Score (Perfect): {score_perfect:.4f}")
    print(f"  Score (Clipped Error): {score_bad:.4f}")

    assert score_perfect > score_bad, "Perfect score should be higher than bad score"
    print("  Metric Logic Verified.")

    # 6. Training Loop Demonstration
    print("\n[5/5] Running Training Loop (Demo)...")

    # Create validation loader (reuse train subset for demo purposes)
    val_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize Trainer with our small loaders
    trainer = Trainer(loader, val_loader)

    # Run Fit
    # This will run for Config.EPOCHS (set to 1)
    trainer.fit()

    # 7. Inference & Inverse Transform Demo
    print("\n--- Inference & Inverse Transform Demo ---")

    # Take the predictions from the earlier manual forward pass
    # preds is (B, 2) -> [mu_scaled, sigma_scaled]
    mu_scaled = preds[:, 0].detach().cpu().numpy()
    sigma_scaled = preds[:, 1].detach().cpu().numpy()

    # Inverse Transform
    mu_final = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
    sigma_final = sigma_scaled * Config.TARGET_STD

    for i in range(len(mu_final)):
        print(
            f"  Sample {i}: Predicted FVC = {mu_final[i]:.2f} ml, Confidence = {sigma_final[i]:.2f} ml"
        )

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    demo_pipeline()
