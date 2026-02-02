import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import sys

# Import from the provided library
from library.config import Config
from library.dataset import create_dataloaders
from library.model import HierarchicalAttentionResNet
from library.train import train_one_epoch, validate
from library.utils import load_category_hierarchy


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("==== Starting Pipeline Demonstration ====")
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 1. Configuration Override
    # ---------------------------------------------------------
    # We modify the Config class attributes directly to enable a fast debug run.
    print("[1/6] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Main process only (avoids overhead/complexity)
    Config.NUM_EPOCHS = 1
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"      Device: {Config.DEVICE}")
    print(f"      Debug Mode: {Config.DEBUG}")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("[2/6] Verifying DataLoaders and Batch Structure...")
    train_loader, val_loader, test_loader = create_dataloaders()

    # Verify dataset length matches debug size
    assert (
        len(train_loader.dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_loader.dataset)}"

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    images = batch["images"]
    batch_index = batch["batch_index"]
    l3_target = batch["l3_target"]

    # Verify Image Tensor: (Total_Images, 3, H, W)
    # Note: Total_Images >= Batch_Size because each product has 1-4 images
    assert images.dim() == 4, f"Images tensor must be 4D, got {images.dim()}"
    assert images.shape[1:] == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {images.shape[1:]}"

    # Verify Batch Index: (Total_Images,)
    assert batch_index.dim() == 1, "Batch index must be 1D"
    assert (
        batch_index.shape[0] == images.shape[0]
    ), "Batch index length must match number of images"

    # Verify Targets: (Batch_Size,)
    assert (
        l3_target.shape[0] == Config.BATCH_SIZE
    ), f"Target batch size mismatch. Expected {Config.BATCH_SIZE}, got {l3_target.shape[0]}"

    print("      Data loading checks passed.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("[3/6] Verifying Model Forward Pass...")
    device = torch.device(Config.DEVICE)
    model = HierarchicalAttentionResNet().to(device)

    # Move batch to device
    images = images.to(device)
    batch_index = batch_index.to(device)

    # Perform forward pass
    outputs = model(images, batch_index)

    # Check existence of hierarchical outputs
    for level in ["logits_l1", "logits_l2", "logits_l3"]:
        assert level in outputs, f"Model output missing {level}"

    # Check shape of the main target level (L3)
    expected_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES_L3)
    assert (
        outputs["logits_l3"].shape == expected_shape
    ), f"L3 Logits shape mismatch. Expected {expected_shape}, got {outputs['logits_l3'].shape}"

    print("      Model forward pass successful.")

    # ---------------------------------------------------------
    # 4. Training Loop Verification
    # ---------------------------------------------------------
    print("[4/6] Verifying Training Step...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Define loss functions
    criteria = {
        "l1": nn.CrossEntropyLoss(),
        "l2": nn.CrossEntropyLoss(),
        "l3": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
    }

    # Run one training epoch
    # We pass None for scheduler to keep it simple
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, None, criteria, device, epoch=0
    )

    # Assertions
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0.0 <= train_acc <= 1.0, f"Training accuracy {train_acc} out of range [0, 1]"

    print(f"      Training step verified. Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # ---------------------------------------------------------
    # 5. Validation Loop Verification
    # ---------------------------------------------------------
    print("[5/6] Verifying Validation Step...")

    val_loss, val_acc = validate(model, val_loader, criteria, device)

    assert not np.isnan(val_loss), "Validation loss is NaN"
    print(f"      Validation verified. Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # ---------------------------------------------------------
    # 6. Inference and Submission Verification
    # ---------------------------------------------------------
    print("[6/6] Verifying Inference and Submission Generation...")

    # Load hierarchy to map integer predictions back to category IDs
    df_hierarchy = load_category_hierarchy(load_cached_data=True)
    df_hierarchy["l3_idx"] = df_hierarchy["l3_idx"].astype(int)
    idx_to_cat = pd.Series(
        df_hierarchy.index.values, index=df_hierarchy["l3_idx"]
    ).to_dict()

    model.eval()
    results = []

    # Custom inference loop to avoid progress bars
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            batch_index = batch["batch_index"].to(device)
            sample_ids = batch["sample_ids"].cpu().numpy()

            outputs = model(images, batch_index)

            # Get L3 predictions (Fine-grained category)
            preds_l3 = outputs["logits_l3"].argmax(dim=1).cpu().numpy()

            for sid, pred_idx in zip(sample_ids, preds_l3):
                cat_id = idx_to_cat.get(pred_idx, -1)
                results.append({"_id": sid, "category_id": cat_id})

    # Create Submission DataFrame
    df_sub = pd.DataFrame(results)

    # Verify Submission Structure
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission rows {len(df_sub)} != Test Set Size {Config.DEBUG_SAMPLE_SIZE}"

    assert (
        "_id" in df_sub.columns and "category_id" in df_sub.columns
    ), "Submission DataFrame missing required columns"

    assert pd.api.types.is_integer_dtype(df_sub["_id"]), "_id must be integer"
    assert pd.api.types.is_integer_dtype(
        df_sub["category_id"]
    ), "category_id must be integer"

    # Save to working directory
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    df_sub.to_csv(output_path, index=False)

    print(f"      Submission file generated at: {output_path}")
    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
