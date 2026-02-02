import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data_processing import load_data
from library.model import InsultDetector
from library.awp import AWP
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Override
    # We override specific Config attributes to make this demo run fast
    print("\n[1] Setting up configuration and seeds...")
    seed_everything(Config.seed)

    # Override Config for speed
    Config.epochs = 1
    Config.batch_size = 4
    Config.n_folds = 2
    Config.debug = True
    # We will use a small subset size for the demo
    DEMO_SUBSET_SIZE = 16

    print(f"    Device: {Config.device}")
    print(f"    Batch Size: {Config.batch_size}")
    print(f"    Subset Size: {DEMO_SUBSET_SIZE}")

    # 2. Data Loading and Processing
    print("\n[2] Loading Data and Generating Features...")
    # This will generate SVD features and return PyTorch Datasets
    # Note: The first run might take a few seconds to fit TF-IDF/SVD
    train_ds, val_ds, test_ds = load_data(load_cached_data=True)

    # Verify Dataset Integrity
    print("    Verifying dataset outputs...")
    sample_item = train_ds[0]
    required_keys = ["input_ids", "attention_mask", "svd_features", "label"]
    for key in required_keys:
        if key not in sample_item:
            raise AssertionError(f"Dataset item missing key: {key}")

    # Verify Shapes
    assert sample_item["input_ids"].shape == (
        Config.max_len,
    ), f"Incorrect input_ids shape: {sample_item['input_ids'].shape}"
    assert sample_item["svd_features"].shape == (
        Config.svd_components,
    ), f"Incorrect svd_features shape: {sample_item['svd_features'].shape}"

    print("    Dataset verification passed.")

    # Create Subsets for Speed
    train_subset = Subset(train_ds, indices=range(DEMO_SUBSET_SIZE))
    val_subset = Subset(val_ds, indices=range(DEMO_SUBSET_SIZE))
    test_subset = Subset(test_ds, indices=range(DEMO_SUBSET_SIZE))

    train_loader = DataLoader(train_subset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=Config.batch_size, shuffle=False)

    # 3. Model Initialization and Forward Pass Check
    print("\n[3] Initializing Model and Checking Forward Pass...")
    model = InsultDetector()
    model.to(Config.device)
    model.eval()

    # Get a batch from loader
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(Config.device)
    mask = batch["attention_mask"].to(Config.device)
    svd = batch["svd_features"].to(Config.device)

    with torch.no_grad():
        output = model(input_ids, mask, svd)

    # Verify Output Shape: (batch_size, num_classes) -> (4, 1)
    assert output.shape == (
        Config.batch_size,
        1,
    ), f"Model output shape mismatch. Expected ({Config.batch_size}, 1), got {output.shape}"

    print(f"    Forward pass successful. Output shape: {output.shape}")

    # 4. AWP Logic Verification
    print("\n[4] Verifying Adversarial Weight Perturbation (AWP)...")
    # We need a model with gradients for AWP, so we switch to train mode and do a dummy backward
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Forward & Backward to populate gradients
    output = model(input_ids, mask, svd)
    loss = torch.nn.BCEWithLogitsLoss()(
        output, batch["label"].to(Config.device).unsqueeze(1)
    )
    loss.backward()

    # Initialize AWP
    awp = AWP(
        model, optimizer, adv_lr=0.1, adv_eps=0.1
    )  # High LR to ensure visible change

    # Pick a parameter to monitor (e.g., classifier weight)
    param_name = "classifier.weight"
    original_weight = model.classifier.weight.data.clone()

    # Attack
    awp.attack()
    perturbed_weight = model.classifier.weight.data

    # Assert weights changed
    diff = torch.norm(perturbed_weight - original_weight).item()
    assert diff > 0, "AWP Attack failed: Weights did not change."
    print(f"    AWP Attack successful. Weight perturbation norm: {diff:.6f}")

    # Restore
    awp.restore()
    restored_weight = model.classifier.weight.data

    # Assert weights restored
    restore_diff = torch.norm(restored_weight - original_weight).item()
    assert restore_diff < 1e-6, f"AWP Restore failed: Weights differ by {restore_diff}"
    print("    AWP Restore successful.")

    # Clear gradients
    optimizer.zero_grad()

    # 5. Trainer Execution (Training Loop)
    print("\n[5] Running Trainer (1 Epoch on Subset)...")
    # Re-initialize model to start fresh
    model = InsultDetector()
    model.to(Config.device)

    trainer = Trainer(model, train_loader, val_loader, Config.device)

    # Run training epoch
    train_loss = trainer.train_epoch(epoch=0)
    print(f"    Training Epoch 0 complete. Loss: {train_loss:.4f}")

    # Validate loss is a valid number
    assert not np.isnan(train_loss), "Training loss is NaN."

    # Run validation epoch
    val_loss, val_auc = trainer.valid_epoch()
    print(f"    Validation complete. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN."
    # AUC might be 0.0 or 1.0 or anything in between depending on the small random batch,
    # but it should be a float.
    assert isinstance(val_auc, float), "Validation AUC is not a float."

    # 6. Inference and Submission Generation
    print("\n[6] Testing Inference and Submission Format...")

    # Manual prediction loop using the trained model on test subset
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(Config.device)
            mask = batch["attention_mask"].to(Config.device)
            svd = batch["svd_features"].to(Config.device)

            logits = model(input_ids, mask, svd)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)

    preds = np.concatenate(preds)

    # Verify predictions
    assert (
        len(preds) == DEMO_SUBSET_SIZE
    ), f"Prediction count mismatch. Expected {DEMO_SUBSET_SIZE}, got {len(preds)}"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions contain values outside [0, 1] range."

    print(f"    Generated {len(preds)} predictions.")
    print(f"    Sample predictions: {preds[:3].flatten()}")

    # Verify metric calculation function
    print("\n[7] Verifying Metric Calculation...")
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    score = calculate_metric(y_true, y_pred)
    print(f"    Test Metric Score (AUC): {score:.4f}")
    assert 0 <= score <= 1, "Metric score out of range."

    print("\n=== Demonstration Complete: All checks passed successfully. ===")


if __name__ == "__main__":
    main()
