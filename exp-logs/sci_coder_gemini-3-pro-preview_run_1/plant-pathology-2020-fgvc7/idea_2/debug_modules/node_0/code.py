import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import (
    AppleDataset,
    get_transforms,
    load_full_train_data,
    load_test_data,
)
from library.model import AppleResNet34
from library.engine import fit, get_weighted_criterion


def run_demo():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    print("==== Starting Apple Disease Detection Demo ====")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Override Config for rapid demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead for small data

    # Ensure directories exist
    Config.setup_directories()

    device = Config.DEVICE
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("\n[Data] Loading and preparing datasets...")

    # Load full training metadata
    full_df = load_full_train_data()

    # Subset data for speed (50 samples total)
    subset_df = full_df.head(50).copy()

    # Split into train (40) and validation (10)
    train_df = subset_df.iloc[:40].reset_index(drop=True)
    val_df = subset_df.iloc[40:].reset_index(drop=True)

    print(f"Train subset size: {len(train_df)}")
    print(f"Val subset size: {len(val_df)}")

    # Instantiate Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"), mode="train")

    # Verify Dataset Logic
    sample_img, sample_label = train_dataset[0]
    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[Model] Initializing AppleResNet34...")

    model = AppleResNet34(pretrained=True)
    model.to(device)

    # Verify Model Output Shape
    dummy_batch = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_batch)

    assert dummy_out.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {dummy_out.shape}"
    print("Model shape verification passed.")

    # ==========================================
    # 4. Training Execution
    # ==========================================
    print("\n[Training] Starting training loop...")

    # Setup Criterion and Optimizer
    criterion = get_weighted_criterion(train_df, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    save_path = os.path.join(Config.MODELS_DIR, "demo_best_model.pth")

    # Run Training
    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=save_path,
    )

    print(f"Training finished. Best Validation AUC: {best_auc:.4f}")

    # Verify model was saved
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model file not found at {save_path}")

    # ==========================================
    # 5. Inference & Validation
    # ==========================================
    print("\n[Inference] Running prediction on test subset...")

    # Load Test Data (Subset)
    test_df = load_test_data().head(10)
    test_dataset = AppleDataset(
        test_df, transforms=get_transforms("valid"), mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Best Model
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    final_probs = np.concatenate(all_probs, axis=0)

    # Verify Predictions
    assert len(final_probs) == len(test_df), "Prediction count mismatch"
    assert (
        final_probs.shape[1] == Config.NUM_CLASSES
    ), "Class count mismatch in predictions"

    # Verify Probabilities sum to 1
    row_sums = final_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    # Format Submission
    submission = pd.DataFrame(final_probs, columns=Config.CLASS_LABELS)
    submission.insert(0, "image_id", all_ids)

    print("Sample Predictions:")
    print(submission.head())

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
