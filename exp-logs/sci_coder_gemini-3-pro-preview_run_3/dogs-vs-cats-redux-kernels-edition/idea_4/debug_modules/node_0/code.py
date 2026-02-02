import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.dataset import load_data, get_transforms, CatDogDataset
from library.models import build_model
from library.engine import set_seed, train_model, predict_with_tta


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # --- 1. Configuration Overrides for Speed ---
    # We modify the Config class attributes directly to run a fast demo.
    print("[Config] Overriding configuration for fast execution...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.N_FOLDS = 2  # Not used in this linear script, but good to note
    Config.USE_TTA = True

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"[Config] Device: {Config.DEVICE}")
    print(f"[Config] Batch Size: {Config.BATCH_SIZE}")
    print(f"[Config] Epochs: {Config.EPOCHS}")

    # --- 2. Data Loading & Dataset Verification ---
    print("\n[Data] Loading metadata and creating datasets...")

    # Load training metadata
    # We force load_cached_data=False to ensure we read the provided CSVs
    full_train_df = load_data(mode="train", load_cached_data=False)

    # Create a tiny subset (32 samples) for demonstration
    subset_indices = np.random.choice(len(full_train_df), 32, replace=False)
    mini_df = full_train_df.iloc[subset_indices].reset_index(drop=True)
    print(f"[Data] Created mini training subset with {len(mini_df)} samples.")

    # Split into train/val for the demo
    train_sub = mini_df.iloc[:24]
    val_sub = mini_df.iloc[24:]

    # Instantiate Datasets
    # Train uses augmentation, Valid uses deterministic resize
    train_ds = CatDogDataset(train_sub, transforms=get_transforms("train"))
    val_ds = CatDogDataset(val_sub, transforms=get_transforms("valid"))

    # Verify a single item
    img, label, img_id = train_ds[0]
    print(
        f"[Data] Sample Image Shape: {img.shape} (Expected: 3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE})"
    )
    print(f"[Data] Sample Label: {label}")

    # Assertions
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image dimensions"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert label.dtype == torch.float32, "Label should be float32 for BCE loss"

    # Create DataLoaders
    # num_workers=0 for simple sequential processing in demo
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- 3. Model Construction ---
    print("\n[Model] Building model...")
    # Using 'convnext_small' as defined in Config.MODEL_BACKBONES
    backbone_name = "convnext_small"
    model = build_model(backbone_name, pretrained=True)
    model.to(Config.DEVICE)
    print(f"[Model] {backbone_name} instantiated and moved to {Config.DEVICE}.")

    # Verify Forward Pass
    dummy_batch = img.unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_batch)

    print(f"[Model] Forward pass output shape: {output.shape}")
    assert output.shape == (
        1,
        1,
    ), "Model output should be (Batch_Size, 1) for binary classification"

    # --- 4. Training Loop Execution ---
    print("\n[Engine] Starting training loop (1 Epoch)...")

    # train_model handles optimizer, scheduler, and saving the best checkpoint
    trained_model, best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=Config.DEVICE,
        fold_idx=0,
        backbone_name=backbone_name,
    )

    print(f"[Engine] Training finished. Best Validation Loss: {best_loss:.6f}")
    assert best_loss < float("inf"), "Training failed to return a valid loss"

    # Verify checkpoint existence
    expected_ckpt = os.path.join(Config.WORKING_DIR, f"{backbone_name}_fold_0.pth")
    if os.path.exists(expected_ckpt):
        print(f"[Engine] Checkpoint verified at: {expected_ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {expected_ckpt}")

    # --- 5. Inference & Submission Generation ---
    print("\n[Inference] Running prediction on test subset...")

    # Load test metadata
    test_df = load_data(mode="test", load_cached_data=False)

    # Subset test data
    mini_test_df = test_df.iloc[:10].reset_index(drop=True)

    # Test Dataset & Loader
    test_ds = CatDogDataset(
        mini_test_df, transforms=get_transforms("valid")
    )  # No augmentation for test
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Predict with TTA
    probs, ids = predict_with_tta(trained_model, test_loader, Config.DEVICE)

    print(f"[Inference] Generated {len(probs)} predictions.")
    print(f"[Inference] Sample Probabilities: {probs[:3]}")
    print(f"[Inference] Sample IDs: {ids[:3]}")

    # Assertions
    assert len(probs) == 10, "Incorrect number of predictions"
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # Simulate Submission File Creation
    submission_df = pd.DataFrame({"id": ids, "label": probs})
    print("\n[Submission] Sample submission dataframe:")
    print(submission_df.head(3))

    print("\n=== Demonstration Complete: All checks passed ===")


if __name__ == "__main__":
    main()
