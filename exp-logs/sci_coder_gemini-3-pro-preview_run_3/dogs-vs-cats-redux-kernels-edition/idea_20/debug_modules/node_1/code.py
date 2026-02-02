import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import (
    load_train_metadata,
    load_test_metadata,
    get_transforms,
    DogCatDataset,
)
from library.models import get_model
from library.engine import train_model, predict
from library.ensemble import find_optimal_weights, weighted_average


def run_demo():
    print(">>> Starting Library Demonstration <<<")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Modify Config for a lightweight demo run
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    seed_everything(42)
    device = get_device()
    print(f"Device selected: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Metadata and Creating Datasets...")

    # Load metadata (force reload from CSVs to ignore any existing cache)
    train_meta = load_train_metadata(load_cached_data=False)
    test_meta = load_test_metadata(load_cached_data=False)

    # Create small subsets for speed (50 train, 20 val, 20 test)
    train_subset = train_meta.iloc[:50].copy().reset_index(drop=True)
    val_subset = train_meta.iloc[50:70].copy().reset_index(drop=True)
    test_subset = test_meta.iloc[:20].copy().reset_index(drop=True)

    print(
        f"Subset sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # Get model config for resolution
    model_key = "resnet50"
    model_cfg = Config.MODEL_CONFIGS[model_key]
    img_size = model_cfg["img_size"]

    # Create Transforms
    train_tfm = get_transforms(img_size, mode="train")
    val_tfm = get_transforms(img_size, mode="val")

    # Instantiate Datasets
    # Note: Val dataset uses 'train' mode to return labels for evaluation
    train_ds = DogCatDataset(train_subset, transform=train_tfm, mode="train")
    val_ds = DogCatDataset(val_subset, transform=val_tfm, mode="train")
    test_ds = DogCatDataset(test_subset, transform=val_tfm, mode="test")

    # Validation: Check dataset output
    sample_img, sample_label = train_ds[0]
    assert sample_img.shape == (3, img_size, img_size), "Incorrect image tensor shape"
    assert isinstance(sample_label, torch.Tensor), "Label must be a tensor"
    print("Dataset integrity check passed.")

    # Create DataLoaders
    # Use num_workers=0 to avoid multiprocessing overhead in short demo
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print(f"\n[Step 2] Initializing Model: {model_key}...")
    model = get_model(model_key, pretrained=True, num_classes=1)
    model = model.to(device)

    # -------------------------------------------------------------------------
    # 4. Training & Evaluation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=1
    )

    # Train
    model, best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=1,
        patience=1,
    )
    print(f"Training finished. Best Validation Loss: {best_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Inference on Test Set...")
    ids, probs = predict(model, test_loader, device)

    # Validation
    assert len(ids) == len(test_subset), "Prediction count mismatch"
    assert len(probs) == len(test_subset), "Probability count mismatch"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of bounds"

    # Create submission dataframe
    sub_df = pd.DataFrame({"id": ids, "label": probs})
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Sample submission saved to {sub_path}")
    print(sub_df.head())

    # -------------------------------------------------------------------------
    # 6. Ensemble Logic
    # -------------------------------------------------------------------------
    print("\n[Step 5] Demonstrating Ensemble Optimization...")

    # Generate synthetic data for ensemble demo
    num_samples = 100
    y_true = np.random.randint(0, 2, size=num_samples)

    # Create synthetic predictions for 3 models
    # Model A: Good model
    noise_a = np.random.normal(0, 0.2, size=num_samples)
    preds_a = np.clip(y_true * 0.8 + 0.1 + noise_a, 0.01, 0.99)

    # Model B: Weak/Random model
    preds_b = np.random.uniform(0.3, 0.7, size=num_samples)

    # Model C: Another Good model
    noise_c = np.random.normal(0, 0.25, size=num_samples)
    preds_c = np.clip(y_true * 0.75 + 0.125 + noise_c, 0.01, 0.99)

    preds_list = [preds_a, preds_b, preds_c]

    # Find weights
    weights = find_optimal_weights(preds_list, y_true)
    print("Optimal Weights found:", weights)

    # Validate weights
    assert len(weights) == 3
    assert np.abs(np.sum(weights) - 1.0) < 1e-6, "Weights do not sum to 1"

    # Compute weighted average
    final_ensemble_preds = weighted_average(preds_list, weights)
    print(f"Ensemble predictions computed (Mean: {np.mean(final_ensemble_preds):.4f})")

    print("\n>>> Demonstration Complete <<<")


if __name__ == "__main__":
    run_demo()
