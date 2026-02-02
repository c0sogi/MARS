import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model_lib
import library.loss as loss_lib
import library.engine as engine
import library.inference as inference


def main():
    print("=== Starting Thoracic Disease Detection Demo ===")

    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    # Ensure working directories exist
    config.setup_directories()

    # 2. Data Preparation (Subset for Speed)
    print("\n[Data] Preparing data subsets...")

    # Load full metadata
    df_train_full = pd.read_csv(config.TRAIN_META_PATH)
    df_val_full = pd.read_csv(config.VAL_META_PATH)
    df_test_full = pd.read_csv(config.TEST_META_PATH)

    # Create small subsets (e.g., 32 train, 8 val, 8 test)
    # We group by image_id to ensure we don't split objects of the same image
    train_imgs = df_train_full["image_id"].unique()[:32]
    val_imgs = df_val_full["image_id"].unique()[:8]
    test_imgs = df_test_full["image_id"].unique()[:8]

    df_train_sub = df_train_full[df_train_full["image_id"].isin(train_imgs)].copy()
    df_val_sub = df_val_full[df_val_full["image_id"].isin(val_imgs)].copy()
    df_test_sub = df_test_full[df_test_full["image_id"].isin(test_imgs)].copy()

    print(
        f"Subset sizes -> Train: {len(df_train_sub)}, Val: {len(df_val_sub)}, Test: {len(df_test_sub)}"
    )

    # Instantiate Datasets directly
    # Note: We use the cache directory from config to store processed .npy files
    train_ds = data.ThoracicDataset(
        df_train_sub, mode="train", cache_dir=config.CACHE_DIR
    )
    val_ds = data.ThoracicDataset(df_val_sub, mode="val", cache_dir=config.CACHE_DIR)
    test_ds = data.ThoracicDataset(df_test_sub, mode="test", cache_dir=config.CACHE_DIR)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=4,  # Small batch size for demo
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # 3. Model Instantiation & Verification
    print("\n[Model] Instantiating EfficientDetDecoupled...")
    model = model_lib.EfficientDetDecoupled(num_classes=config.NUM_CLASSES)
    model.to(device)

    # Get a single batch to verify shapes
    images, targets, _, _ = next(iter(train_loader))
    images = images.to(device)

    print(f"Input Image Shape: {images.shape}")

    # Forward pass
    outputs = model(images)

    # Expected output shapes (Stride 4)
    expected_feat_size = config.IMG_SIZE // 4

    # Assertions
    assert outputs["heatmap"].shape == (
        4,
        config.NUM_CLASSES,
        expected_feat_size,
        expected_feat_size,
    ), f"Heatmap shape mismatch. Got {outputs['heatmap'].shape}"
    assert outputs["size"].shape == (
        4,
        2,
        expected_feat_size,
        expected_feat_size,
    ), f"Size head shape mismatch. Got {outputs['size'].shape}"
    assert outputs["offset"].shape == (
        4,
        2,
        expected_feat_size,
        expected_feat_size,
    ), f"Offset head shape mismatch. Got {outputs['offset'].shape}"
    assert outputs["global_logits"].shape == (
        4,
        1,
    ), f"Global head shape mismatch. Got {outputs['global_logits'].shape}"

    print("Model output shapes verified successfully.")

    # 4. Loss Verification
    print("\n[Loss] Verifying ThoracicLoss...")
    criterion = loss_lib.ThoracicLoss()

    # Move targets to device
    targets = {k: v.to(device) for k, v in targets.items()}

    loss, loss_dict = criterion(outputs, targets)

    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Loss Components: {loss_dict}")

    assert torch.isfinite(loss), "Loss is not finite!"
    assert "hm_loss" in loss_dict
    assert "size_loss" in loss_dict

    # 5. Training Loop Execution
    print("\n[Training] Running 1 Epoch on subset...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simple scheduler for demo
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Run fit
    engine.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=1,
        device=device,
        save_path=config.CHECKPOINT_DIR,
    )

    # Verify checkpoint creation
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created!"
    print("Training loop completed and checkpoint saved.")

    # 6. Inference
    print("\n[Inference] Generating predictions on test subset...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Run prediction
    predictions = inference.predict_and_format(
        model=model,
        data_loader=test_loader,
        device=device,
        threshold=0.2,
        gate_threshold=0.8,
    )

    # Verify predictions
    assert len(predictions) > 0, "No predictions generated."
    assert len(predictions[0]) == 2, "Prediction row format incorrect."

    # Create DataFrame
    df_sub = pd.DataFrame(predictions, columns=["image_id", "PredictionString"])
    print("\nSample Predictions:")
    print(df_sub.head())

    # Save submission (optional, just to demonstrate full pipeline)
    sub_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
