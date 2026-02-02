import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch.nn as nn

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.features import SVDFeatureExtractor, get_fold_features
from library.dataset import prepare_fold_data, InsultDataset
from library.model import HybridModel
from library.awp import AWP
from library.trainer import (
    train_one_epoch,
    valid_one_epoch,
    predict,
    get_optimizer_params,
)


def main():
    print("=== Starting Demonstration of Insult Detection Pipeline ===")

    # 1. Setup & Configuration Overrides for Speed
    print("\n[Step 1] Configuring environment...")
    seed_everything(Config.seed)

    # Override Config for rapid execution
    Config.debug = True
    Config.debug_sample_size = 50  # Small subset for demo
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.svd_components = 16  # Reduced components for speed/memory in demo

    # Ensure working directories exist (handled by Config import, but good to double check)
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(Config.model_dir, exist_ok=True)

    # 2. Data Preparation & Feature Engineering
    print("\n[Step 2] Preparing Data and Features...")

    # Initialize Tokenizer
    # We use the tokenizer associated with the model backbone defined in Config
    tokenizer = AutoTokenizer.from_pretrained(Config.model_a_name)

    # Prepare data for Fold 0
    # This handles loading CSVs, splitting, SVD feature generation, and Dataset creation
    train_dataset, val_dataset, test_dataset = prepare_fold_data(
        fold=0,
        tokenizer=tokenizer,
        load_cached_data=False,  # Force re-computation for demonstration
    )

    # Verification
    print(f"  Train Dataset Size: {len(train_dataset)}")
    print(f"  Val Dataset Size: {len(val_dataset)}")
    print(f"  Test Dataset Size: {len(test_dataset)}")

    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(val_dataset) > 0, "Validation dataset is empty"

    # Check a single sample
    sample = train_dataset[0]
    required_keys = ["input_ids", "attention_mask", "svd_features", "target"]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset sample"

    print("  Dataset verification passed.")

    # 3. Model Initialization
    print("\n[Step 3] Initializing Hybrid Model...")
    device = Config.device
    model = HybridModel(Config.model_a_name, pretrained=True)
    model.to(device)

    # Create a dummy batch to verify forward pass
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    svd_features = batch["svd_features"].to(device)
    targets = batch["target"].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        # Note: token_type_ids might not be present for all models (e.g. RoBERTa), handle gracefully
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(input_ids, attention_mask, svd_features, token_type_ids)

    print(f"  Logits Shape: {logits.shape}")
    assert logits.shape == (2, 1), f"Expected output shape (2, 1), got {logits.shape}"
    print("  Model forward pass verified.")

    # 4. AWP Demonstration
    print("\n[Step 4] Demonstrating Adversarial Weight Perturbation (AWP)...")
    # Setup optimizer for AWP initialization
    optimizer_params = get_optimizer_params(model, 1e-5, 1e-3, 0.01)
    optimizer = AdamW(optimizer_params)

    awp = AWP(model, optimizer, adv_lr=0.1, adv_eps=0.01)

    # Simulate attack
    # We need gradients for AWP to work, so we do a quick backward pass
    model.train()
    logits = model(input_ids, attention_mask, svd_features, token_type_ids)
    loss = nn.BCEWithLogitsLoss()(logits.view(-1), targets)
    loss.backward()

    # Save original weight of a specific parameter to verify change
    param_name = list(model.fc.named_parameters())[0][
        0
    ]  # Get first param name of fc layer
    original_weight = model.fc.weight.data.clone()

    print("  Executing AWP attack...")
    awp.attack()
    perturbed_weight = model.fc.weight.data.clone()

    # Verify weights changed
    diff = torch.norm(original_weight - perturbed_weight).item()
    print(f"  Weight perturbation magnitude: {diff:.6f}")
    assert diff > 0, "AWP did not perturb weights"

    print("  Restoring weights...")
    awp.restore()
    restored_weight = model.fc.weight.data

    # Verify restoration
    restore_diff = torch.norm(original_weight - restored_weight).item()
    assert restore_diff < 1e-6, "AWP restore failed"
    print("  AWP logic verified.")

    # Clear gradients
    optimizer.zero_grad()

    # 5. Training Loop Integration
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.valid_batch_size, shuffle=False
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Run Train Epoch
    train_one_epoch(
        epoch=1,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        criterion=criterion,
        awp=awp,
        scaler=scaler,
    )

    # Run Valid Epoch
    print("  Running Validation...")
    val_preds, val_auc = valid_one_epoch(
        epoch=1, model=model, val_loader=val_loader, device=device, criterion=criterion
    )

    print(f"  Validation AUC: {val_auc:.4f}")
    assert len(val_preds) == len(val_dataset), "Validation predictions length mismatch"
    print("  Training loop execution verified.")

    # 6. Inference
    print("\n[Step 6] Running Inference on Test Set...")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.valid_batch_size, shuffle=False
    )

    test_preds = predict(model, test_loader, device)

    print(f"  Test Predictions Shape: {test_preds.shape}")
    assert len(test_preds) == len(test_dataset), "Test predictions length mismatch"
    assert (test_preds >= 0).all() and (
        test_preds <= 1
    ).all(), "Predictions out of probability range [0, 1]"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
