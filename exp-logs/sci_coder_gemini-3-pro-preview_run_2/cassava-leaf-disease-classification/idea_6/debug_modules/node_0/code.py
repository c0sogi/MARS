import os
import shutil
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import Mixup

# Import library modules
from library.config import CFG
from library import utils, data, model, engine


def main():
    print("Starting Cassava Disease Classification Library Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    # Override CFG parameters for a quick demonstration
    CFG.seed = 42
    CFG.epochs = 1
    CFG.batch_size = (
        4  # Small batch size to ensure multiple batches even with small data
    )
    CFG.metadata_dir = "./working/demo_metadata"
    CFG.output_dir = "./working/demo_output"
    CFG.image_size = 224  # Reduce size for speed

    # Ensure reproducibility
    utils.seed_everything(CFG.seed)

    # Clean up previous demo runs if any
    if os.path.exists(CFG.metadata_dir):
        shutil.rmtree(CFG.metadata_dir)
    if os.path.exists(CFG.output_dir):
        shutil.rmtree(CFG.output_dir)

    os.makedirs(CFG.metadata_dir, exist_ok=True)
    os.makedirs(CFG.output_dir, exist_ok=True)

    print(f"Device: {CFG.device}")

    # -------------------------------------------------------------------------
    # 2. Prepare Subset Data
    # -------------------------------------------------------------------------
    print("\n[Data] Preparing subset metadata for speed...")

    # Load original metadata
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    df_train_orig = pd.read_csv(orig_train_path)
    df_val_orig = pd.read_csv(orig_val_path)
    df_test_orig = pd.read_csv(orig_test_path)

    # Create small subsets (enough for a few batches)
    # We take 20 train, 10 val, 10 test samples
    df_train_sub = df_train_orig.head(20).copy()
    df_val_sub = df_val_orig.head(10).copy()
    df_test_sub = df_test_orig.head(10).copy()

    # Save to the new temporary metadata directory
    df_train_sub.to_csv(os.path.join(CFG.metadata_dir, "train.csv"), index=False)
    df_val_sub.to_csv(os.path.join(CFG.metadata_dir, "val.csv"), index=False)
    df_test_sub.to_csv(os.path.join(CFG.metadata_dir, "test.csv"), index=False)

    print(
        f"Subset sizes - Train: {len(df_train_sub)}, Val: {len(df_val_sub)}, Test: {len(df_test_sub)}"
    )

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[Data] Verifying DataLoaders and Folds...")

    # Test prepare_folds logic
    # This should combine train/val subsets and create folds
    df_folds = data.prepare_folds(load_cached_data=False)

    # Assertions
    assert len(df_folds) == len(df_train_sub) + len(
        df_val_sub
    ), "Fold dataframe size mismatch"
    assert "fold" in df_folds.columns, "Fold column missing"
    print("prepare_folds: Success")

    # Test get_loaders
    # We use fold 0. With 30 samples total and 5 folds, fold 0 should have ~6 validation samples.
    train_loader, val_loader = data.get_loaders(fold=0, load_cached_data=True)

    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Val Loader: {len(val_loader)} batches")

    # Fetch a batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Shapes - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.image_size,
        CFG.image_size,
    ), "Image batch shape incorrect"
    assert labels.shape == (CFG.batch_size,), "Label batch shape incorrect"
    print("DataLoaders: Success")

    # -------------------------------------------------------------------------
    # 4. Verify Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[Model] Instantiating CassavaClassifier...")

    # Use pretrained=False to avoid downloading weights during the demo and speed up initialization
    model_instance = model.CassavaClassifier(pretrained=False)
    model_instance.to(CFG.device)

    # Verify forward pass
    with torch.no_grad():
        # Use the batch fetched earlier
        images = images.to(CFG.device)
        output = model_instance(images)

    print(f"Output Logits Shape: {output.shape}")
    assert output.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Model output shape incorrect"
    print("Model Instantiation: Success")

    # -------------------------------------------------------------------------
    # 5. Verify Training Engine
    # -------------------------------------------------------------------------
    print("\n[Engine] Verifying Training Loop...")

    # Setup Optimizer and Scaler
    optimizer = optim.AdamW(model_instance.parameters(), lr=CFG.lr)
    # Using torch.amp.GradScaler for modern PyTorch compatibility
    scaler = torch.amp.GradScaler("cuda")

    # Setup Mixup (as used in config)
    mixup_fn = Mixup(
        mixup_alpha=CFG.mixup_alpha,
        cutmix_alpha=CFG.cutmix_alpha,
        prob=CFG.mixup_prob,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=CFG.num_classes,
    )

    # Run one epoch of training
    # Note: This uses the subset loader, so it will be very fast
    train_loss = engine.train_one_epoch(
        epoch=1,
        model=model_instance,
        optimizer=optimizer,
        data_loader=train_loader,
        device=CFG.device,
        model_ema=None,  # Skipping EMA for simple demo
        mixup_fn=mixup_fn,
        scaler=scaler,
    )

    print(f"Train Loss: {train_loss:.4f}")
    assert not math.isnan(train_loss), "Training loss is NaN"
    print("Training Loop: Success")

    # -------------------------------------------------------------------------
    # 6. Verify Validation Engine
    # -------------------------------------------------------------------------
    print("\n[Engine] Verifying Validation Loop...")

    val_loss, val_acc = engine.valid_one_epoch(
        model=model_instance, data_loader=val_loader, device=CFG.device
    )

    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    assert not math.isnan(val_loss), "Validation loss is NaN"
    print("Validation Loop: Success")

    # -------------------------------------------------------------------------
    # 7. Verify Prediction/Inference
    # -------------------------------------------------------------------------
    print("\n[Engine] Verifying Inference...")

    test_loader = data.get_test_loader()

    # Run prediction
    predictions = engine.predict(model_instance, test_loader, CFG.device)

    print(f"Predictions Shape: {predictions.shape}")

    # Check shape against test subset size
    assert predictions.shape == (
        len(df_test_sub),
        CFG.num_classes,
    ), "Prediction shape mismatch"

    # Check values are probabilities (sum to ~1)
    sums = predictions.sum(dim=1)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-5
    ), "Predictions are not valid probabilities"

    print("Inference: Success")

    print("\n=======================================")
    print("      ALL CHECKS PASSED SUCCESSFULLY     ")
    print("=======================================")


if __name__ == "__main__":
    main()
