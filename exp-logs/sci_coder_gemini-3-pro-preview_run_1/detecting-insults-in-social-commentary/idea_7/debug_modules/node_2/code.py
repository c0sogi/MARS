import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shutil
import warnings


# ==========================================
# 1. Suppress Output & Warnings
# ==========================================
# Suppress tqdm progress bars used in library modules
class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable or []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


import tqdm.auto

tqdm.auto.tqdm = SilentTqdm

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 2. Import Library Components
# ==========================================
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import load_data, create_dataloader, get_tokenizer
from library.model import HybridDeberta, AWP, train_one_epoch, validate
from library.inference import predict_fn, generate_pseudo_labels


def main():
    print("Starting Insult Detection Library Demo...")

    # ==========================================
    # 3. Setup & Configuration Overrides
    # ==========================================
    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.N_FOLDS = 2
    Config.debug = True  # Forces load_data to use truncated datasets
    Config.SVD_DIM = 64  # Reduce SVD dim to be < 100 samples in debug mode

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # ==========================================
    # 4. Data Loading & Processing Demo
    # ==========================================
    print("\n[Demo] Loading and Processing Data...")

    # Load data in debug mode (truncated) without using cache to demonstrate processing
    # Note: load_data handles TF-IDF and SVD computation internally
    train_df, val_df, test_df, train_svd, val_svd, test_svd = load_data(
        load_cached_data=False, debug=True
    )

    # Verifications
    print("Verifying data shapes...")
    # Debug mode in library truncates train to 100, val/test to 50
    assert (
        len(train_df) == 100
    ), f"Expected 100 train samples in debug mode, got {len(train_df)}"
    assert (
        len(val_df) == 50
    ), f"Expected 50 val samples in debug mode, got {len(val_df)}"
    assert (
        len(test_df) == 50
    ), f"Expected 50 test samples in debug mode, got {len(test_df)}"

    # Verify SVD features
    assert train_svd.shape == (100, Config.SVD_DIM), "Train SVD shape mismatch"
    assert val_svd.shape == (50, Config.SVD_DIM), "Val SVD shape mismatch"
    assert not np.isnan(train_svd).any(), "SVD features contain NaNs"

    print("Data loading verified successfully.")

    # ==========================================
    # 5. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[Demo] Initializing HybridDeberta Model...")

    model = HybridDeberta(config=Config).to(device)
    tokenizer = get_tokenizer()

    # Create a dummy batch to verify forward pass
    print("Verifying forward pass...")
    dummy_text = ["This is a test comment.", "Another comment."]
    dummy_svd = torch.randn(2, Config.SVD_DIM).to(device)

    encoded = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=Config.MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, dummy_svd)

    assert outputs.shape == (2, 1), f"Expected output shape (2, 1), got {outputs.shape}"
    print("Forward pass verified.")

    # ==========================================
    # 6. Adversarial Weight Perturbation (AWP) Demo
    # ==========================================
    print("\n[Demo] Verifying AWP Logic...")

    model.train()
    # Optimizer is needed for AWP
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    awp = AWP(
        model, optimizer, adv_lr=0.1, adv_eps=1.0
    )  # High LR to ensure visible change

    # We need gradients to perform an attack
    outputs = model(input_ids, attention_mask, dummy_svd)
    loss = outputs.mean()
    loss.backward()

    # Capture original weight of the final layer
    original_weight = model.fc.weight.data.clone()

    # Attack
    awp.attack()
    perturbed_weight = model.fc.weight.data.clone()

    # Verify weights changed
    diff = torch.norm(original_weight - perturbed_weight).item()
    assert diff > 0, "AWP attack did not modify weights"

    # Restore
    awp.restore()
    restored_weight = model.fc.weight.data.clone()

    # Verify restoration
    restore_diff = torch.norm(original_weight - restored_weight).item()
    assert restore_diff < 1e-6, "AWP restore failed to recover original weights"

    optimizer.zero_grad()
    print("AWP logic verified.")

    # ==========================================
    # 7. Training & Validation Loop Demo
    # ==========================================
    print("\n[Demo] Running Training Loop (1 Epoch)...")

    # Create DataLoaders
    train_loader = create_dataloader(
        train_df, train_svd, tokenizer, batch_size=Config.BATCH_SIZE, is_train=True
    )
    val_loader = create_dataloader(
        val_df, val_svd, tokenizer, batch_size=Config.BATCH_SIZE, is_train=False
    )

    criterion = nn.BCEWithLogitsLoss()
    scheduler = None  # Skip scheduler for simple demo

    # Run one epoch
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        epoch=1,
        use_awp=False,
    )

    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run validation
    val_loss, val_auc, val_preds = validate(model, val_loader, criterion, device)

    print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    assert 0 <= val_auc <= 1, "AUC score out of range"
    assert len(val_preds) == len(val_df), "Validation predictions length mismatch"

    # ==========================================
    # 8. Inference & Pseudo-Labeling Demo
    # ==========================================
    print("\n[Demo] Running Inference and Pseudo-Labeling...")

    # Inference
    test_preds = predict_fn(
        model,
        test_df,
        test_svd,
        device,
        batch_size=Config.BATCH_SIZE,
        tokenizer=tokenizer,
    )

    assert len(test_preds) == len(test_df), "Test predictions length mismatch"
    assert (
        test_preds.min() >= 0 and test_preds.max() <= 1
    ), "Predictions out of probability range [0,1]"

    # Pseudo-Labeling
    # Use extreme thresholds to force logic check, or standard ones
    # We'll use the predictions we just generated
    pseudo_df, pseudo_svd = generate_pseudo_labels(
        test_df, test_svd, test_preds, high_thresh=0.8, low_thresh=0.2
    )

    if len(pseudo_df) > 0:
        assert (
            "Insult" in pseudo_df.columns
        ), "Pseudo-labeled dataframe missing target column"
        assert pseudo_svd.shape[0] == len(
            pseudo_df
        ), "Pseudo-label SVD features mismatch"
        print(f"Generated {len(pseudo_df)} pseudo-labels.")
    else:
        print("No pseudo-labels generated (expected with random weights/small data).")
        # Verify empty return structure
        assert "Insult" in pseudo_df.columns
        assert pseudo_svd.shape[0] == 0

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
