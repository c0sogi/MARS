import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, logging as hf_logging

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_cpc_texts, compute_pearson_score
from library.data import prepare_loaders
from library.model import CustomDeberta
from library.awp import AWP
from library.train import train_fn, valid_fn

# Suppress HuggingFace warnings for cleaner output
hf_logging.set_verbosity_error()


def run_demo():
    print("=== Starting Phrase Similarity Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Override Config for speed and demonstration purposes
    Config.debug = True
    Config.debug_sample_size = 64  # Small subset for speed
    Config.epochs = 1
    Config.train_batch_size = 8
    Config.valid_batch_size = 16
    Config.model_name = "microsoft/deberta-v3-xsmall"  # Smaller model for fast demo
    Config.output_dir = "./working/demo_run"
    Config.num_workers = 2
    Config.use_awp = True
    Config.awp_start_epoch = 0  # Enable AWP immediately for demo

    # Ensure output directory exists
    if os.path.exists(Config.output_dir):
        shutil.rmtree(Config.output_dir)
    os.makedirs(Config.output_dir, exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Model: {Config.model_name}")
    print(f"    Output Dir: {Config.output_dir}")

    # ---------------------------------------------------------
    # 2. Utils Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Utils...")

    # Test CPC Text loading
    cpc_texts = get_cpc_texts(Config.cpc_codes_path)
    print(f"    Loaded CPC Texts: {len(cpc_texts)} entries found.")

    # Test Metric
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.1, 0.4, 0.9])
    score = compute_pearson_score(y_true, y_pred)
    print(f"    Test Pearson Score: {score:.4f}")
    assert -1.0 <= score <= 1.0, "Pearson score out of range"

    # ---------------------------------------------------------
    # 3. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Prepare loaders (this will trigger _process_data and caching)
    train_loader, val_loader, test_loader = prepare_loaders(
        tokenizer, load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"    Batch Input Shape: {input_ids.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    # Verify label range (0 to 4 for 5 classes)
    assert labels.max() < 5 and labels.min() >= 0, "Labels out of expected range [0, 4]"

    # ---------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Verifying Model...")

    device = Config.device
    model = CustomDeberta(Config.model_name, pretrained=True)
    model.to(device)

    # Move batch to device
    b_input_ids = input_ids.to(device)
    b_mask = batch["attention_mask"].to(device)
    b_labels = labels.to(device)

    # Forward pass
    logits = model(b_input_ids, b_mask)

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (
        input_ids.shape[0],
        5,
    ), f"Expected shape ({input_ids.shape[0]}, 5), got {logits.shape}"

    # Loss calculation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, b_labels)
    print(f"    Initial Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # ---------------------------------------------------------
    # 5. AWP (Adversarial Weight Perturbation) Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying AWP...")

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scaler = GradScaler()

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=0,
        scaler=scaler,
    )

    # Perform backward pass to populate gradients (needed for AWP)
    scaler.scale(loss).backward()

    # Check weights before attack
    param_check = list(model.parameters())[0].clone()

    # Attack step
    awp.attack_step()

    # Check weights after attack (should be different)
    param_perturbed = list(model.parameters())[0]
    diff = torch.norm(param_check - param_perturbed).item()
    print(f"    Weight perturbation magnitude: {diff:.6f}")
    assert diff > 0, "AWP did not perturb weights"

    # Restore weights
    awp._restore()
    param_restored = list(model.parameters())[0]
    diff_restored = torch.norm(param_check - param_restored).item()
    assert diff_restored == 0, "AWP restore failed"
    print("    AWP Attack and Restore successful.")

    # Clear gradients
    optimizer.zero_grad()

    # ---------------------------------------------------------
    # 6. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Training Function...")

    # Run one epoch of training using the provided train_fn
    # We use a dummy scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    avg_train_loss = train_fn(
        train_loader,
        model,
        criterion,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
        scaler=scaler,
    )

    print(f"    Average Train Loss: {avg_train_loss:.4f}")
    assert avg_train_loss > 0, "Training loss should be positive"

    # ---------------------------------------------------------
    # 7. Validation Loop Verification
    # ---------------------------------------------------------
    print("\n[7] Verifying Validation Function...")

    val_loss, val_preds, val_labels = valid_fn(val_loader, model, criterion, device)

    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Predictions Shape: {val_preds.shape}")
    print(f"    Ground Truth Shape: {val_labels.shape}")

    # Check predictions are in valid score range [0, 1]
    # Note: The model is untrained, so predictions might be slightly off, but should be bounded roughly.
    # The valid_fn calculates weighted sum of [0, 0.25, 0.5, 0.75, 1.0], so result is strictly [0, 1].
    assert (
        val_preds.min() >= 0.0 and val_preds.max() <= 1.0
    ), "Predictions out of range [0, 1]"

    # Compute Score
    val_score = compute_pearson_score(val_labels, val_preds)
    print(f"    Validation Pearson Score: {val_score:.4f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
