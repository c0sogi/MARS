import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import HybridCactusClassifier, CovariancePooling
from library.engine import train_engine, predict_tta


def main():
    print("Starting Cactus Identification Demo...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("\n[1] Overriding Configuration for Fast Demonstration...")
    # Enable debug mode to load only 200 samples
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200

    # Reduce training duration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16  # Smaller batch size for the small debug dataset
    Config.SEEDS = [42]  # Run only one seed

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Reproducibility
    # ==========================================
    print("\n[2] Setting Random Seeds...")
    set_seed(42)

    # ==========================================
    # 3. Data Loading & Verification
    # ==========================================
    print("\n[3] Initializing DataLoaders...")
    dataloaders = get_dataloaders(load_cached_data=False)

    # Verify keys
    assert "train" in dataloaders
    assert "val" in dataloaders
    assert "test" in dataloaders

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Batch Structure
    print("Verifying batch structure...")
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    # Check shapes
    # Expected: (Batch, 3, 32, 32)
    assert images.dim() == 4
    assert images.shape[1] == 3
    assert images.shape[2] == 32
    assert images.shape[3] == 32

    # Expected: (Batch,) or (Batch, 1) depending on dataset implementation
    # The dataset returns labels as float tensors. Engine expects (B, 1) but dataset gives (B,).
    # The engine unsqueezes it. Let's check what dataset gives.
    assert labels.dim() == 1
    assert labels.shape[0] == images.shape[0]

    print("Batch structure verified successfully.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[4] Initializing and Verifying Model...")
    device = Config.DEVICE
    model = HybridCactusClassifier().to(device)

    # Test Covariance Pooling specifically
    print("Testing CovariancePooling logic...")
    # 64 channels, 16x16 feature map
    dummy_feat = torch.randn(2, 64, 16, 16).to(device)
    cov_pool = CovariancePooling(num_features=64).to(device)
    out_cov = cov_pool(dummy_feat)

    # Output dim should be C * (C+1) / 2 = 64 * 65 / 2 = 2080
    expected_dim = int(64 * 65 / 2)
    assert out_cov.shape == (2, expected_dim)
    print(f"CovariancePooling output shape correct: {out_cov.shape}")

    # Test Full Model Forward Pass
    print("Testing full model forward pass...")
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    output = model(dummy_input)

    # Expected output: (Batch, 1) (Logits)
    assert output.shape == (4, 1)
    print(f"Model output shape correct: {output.shape}")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[5] Running Training Loop (Demo)...")

    seed = Config.SEEDS[0]

    # Setup Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Run Engine
    best_auc = train_engine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        seed=seed,
    )

    print(f"Training complete. Best AUC: {best_auc}")

    # Verify Checkpoint Exists
    checkpoint_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
    assert os.path.exists(checkpoint_path)
    print(f"Checkpoint verified at: {checkpoint_path}")

    # ==========================================
    # 6. Inference & TTA Verification
    # ==========================================
    print("\n[6] Running Inference with TTA...")

    # Load best model state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Predict
    df_pred = predict_tta(model, test_loader, device)

    # Verify DataFrame
    print("Verifying submission DataFrame...")
    assert isinstance(df_pred, pd.DataFrame)
    assert "id" in df_pred.columns
    assert "has_cactus" in df_pred.columns

    # Check length (should match debug sample size if test set is also sampled in debug mode)
    # The dataset loader logic for debug slices the input dataframe.
    # So we expect Config.DEBUG_SAMPLE_SIZE rows.
    assert len(df_pred) == Config.DEBUG_SAMPLE_SIZE

    # Check probability range
    probs = df_pred["has_cactus"].values
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    print("Inference output verified.")
    print(df_pred.head())

    # ==========================================
    # 7. Metric Verification
    # ==========================================
    print("\n[7] Verifying Metric Calculation...")
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    score = calculate_roc_auc(y_true, y_scores)
    print(f"Calculated AUC: {score}")
    assert 0.0 <= score <= 1.0

    print("\n==========================================")
    print("       DEMONSTRATION SUCCESSFUL           ")
    print("==========================================")


if __name__ == "__main__":
    main()
